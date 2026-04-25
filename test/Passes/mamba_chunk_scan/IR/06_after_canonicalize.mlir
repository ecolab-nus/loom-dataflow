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
                    %27 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %28 = loom.init_tensor %27[64] : memref<64xf16> -> tensor<64xf16>
                    %29 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %30 = arith.muli %arg11, %c2 : index
                    %31 = arith.addi %arg9, %30 : index
                    %32 = arith.muli %arg12, %c2 : index
                    %33 = arith.muli %arg8, %c4 : index
                    %34 = arith.addi %32, %33 : index
                    %35 = arith.addi %32, %c1 : index
                    %36 = arith.addi %35, %33 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %37 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %38 = loom.sync ins(%37 : tensor<64xf16>) outs(%28 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %26 : memref<64xf16>
                    %39 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %40 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
                    %41 = loom.init_tensor %40[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %42 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
                    %43 = loom.init_tensor %42[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %44 = loom.broadcast ins(%38 : tensor<64xf16>) outs(%43 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %45 = arith.muli %arg12, %c1024 : index
                    %46 = arith.addi %24, %45 : index
                    %47 = arith.divui %22, %c64 : index
                    %48 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %49 = loom.semaphore_take %48 : memref<64x64xf16> -> memref<64x64xf16>
                    %50 = loom.init_tensor %49[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %51 = loom.semaphore_take %48 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %52 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %46, %47, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %53 = loom.bufferize_to_tensor %51[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %54 = loom.sync ins(%53 : tensor<64x64xf16>) outs(%50 : tensor<64x64xf16>) -> tensor<64x64xf16>
                    loom.semaphore_give %51 : memref<64x64xf16>
                    %55 = arith.muli %arg10, %c32 : index
                    %56 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %57 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                    %58 = loom.init_tensor %57[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %59 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %60 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %55)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %61 = arith.addi %30, %c1 : index
                    %62 = arith.addi %arg10, %32 : index
                    %63 = arith.addi %62, %33 : index
                    loom.copy %reinterpret_cast_3, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%30, %63], LR : [%61, %63]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %64 = loom.bufferize_to_tensor %59[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %65 = loom.sync ins(%64 : tensor<64x32xf16>) outs(%58 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %59 : memref<64x32xf16>
                    %66 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %67 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %68 = loom.init_tensor %67[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %69 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %70 = loom.init_tensor %69[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %71 = linalg.fill ins(%cst : f16) outs(%68 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %72 = linalg.matmul ins(%54, %65 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%71 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<64x32xf16>
                    loom.semaphore_give %49 : memref<64x64xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %44 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%70 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %122 = math.exp %in_7 : f16
                      %123 = arith.mulf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %67 : memref<64x32xf16>
                    loom.semaphore_give %42 : memref<64x32xf16>
                    %74 = arith.addi %21, %c1 : index
                    %75 = arith.muli %74, %c64 : index
                    %76 = arith.ceildivui %75, %c64 : index
                    %77 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %78 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %79 = loom.init_tensor %78[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %80 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %81 = loom.init_tensor %80[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %82 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %83 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %84 = loom.init_tensor %83[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %85 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %86 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %87 = loom.init_tensor %86[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %88 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %89 = loom.alloc [64] on @L1 : memref<64xf16>
                    %90 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %91 = loom.init_tensor %90[64] : memref<64xf16> -> tensor<64xf16>
                    %92 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %93 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %94 = loom.init_tensor %93[64] : memref<64xf16> -> tensor<64xf16>
                    %95 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %96 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %97 = loom.semaphore_take %96 : memref<32x64xf16> -> memref<32x64xf16>
                    %98 = loom.init_tensor %97[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %99 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %100 = loom.semaphore_take %99 : memref<32x64xf16> -> memref<32x64xf16>
                    %101 = loom.init_tensor %100[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %102 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %103 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %104 = loom.init_tensor %103[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %105 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %106 = scf.for %arg15 = %c0 to %76 step %c1 iter_args(%arg16 = %73) -> (tensor<64x32xf16>) {
                      %122 = arith.muli %arg15, %c64 : index
                      %123 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %47, %24, %122)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%123], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %124 = loom.bufferize_to_tensor %88[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %125 = loom.sync ins(%124 : tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) -> tensor<64x64xf16>
                      loom.semaphore_give %88 : memref<64x64xf16>
                      %126 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%126], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %95 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %127 = loom.bufferize_to_tensor %95[64] : memref<64xf16> -> tensor<64xf16>
                      %128 = loom.sync ins(%127 : tensor<64xf16>) outs(%94 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %95 : memref<64xf16>
                      %129 = loom.broadcast ins(%38 : tensor<64xf16>) outs(%41 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %130 = loom.broadcast ins(%128 : tensor<64xf16>) outs(%98 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %93 : memref<64xf16>
                      %131 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%131], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %132 = loom.bufferize_to_tensor %92[64] : memref<64xf16> -> tensor<64xf16>
                      %133 = loom.sync ins(%132 : tensor<64xf16>) outs(%91 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %92 : memref<64xf16>
                      %134 = loom.broadcast ins(%133 : tensor<64xf16>) outs(%101 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %90 : memref<64xf16>
                      %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%125, %129, %130, %134 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %143 = arith.subf %in_11, %in_12 : f16
                        %144 = math.exp %143 : f16
                        %145 = arith.mulf %in, %144 : f16
                        %146 = arith.mulf %145, %in_13 : f16
                        linalg.yield %146 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %100 : memref<32x64xf16>
                      loom.semaphore_give %97 : memref<32x64xf16>
                      loom.semaphore_give %40 : memref<64x32xf16>
                      %136 = arith.addi %122, %45 : index
                      %137 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %136, %22, %55)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%137], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %138 = loom.bufferize_to_tensor %105[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %139 = loom.sync ins(%138 : tensor<64x32xf16>) outs(%104 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %105 : memref<64x32xf16>
                      %140 = linalg.fill ins(%cst : f16) outs(%84 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %141 = linalg.matmul ins(%135, %139 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%140 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %103 : memref<64x32xf16>
                      loom.semaphore_give %86 : memref<64x64xf16>
                      %142 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %141 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %143 = arith.addf %in, %in_11 : f16
                        linalg.yield %143 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %83 : memref<64x32xf16>
                      scf.yield %142 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %27 : memref<64xf16>
                    %107 = loom.alloc [1] on @L1 : memref<f16>
                    %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %109 = loom.init_tensor %108[] : memref<f16> -> tensor<f16>
                    %110 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %111 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%111], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %112 = arith.addi %33, %c3 : index
                    loom.copy %reinterpret_cast_4, %110 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %33], LR : [%c7, %112]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %113 = loom.bufferize_to_tensor %110[] : memref<f16> -> tensor<f16>
                    %114 = loom.sync ins(%113 : tensor<f16>) outs(%109 : tensor<f16>) -> tensor<f16>
                    loom.semaphore_give %110 : memref<f16>
                    %115 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %46, %22, %55)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%115], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %116 = loom.bufferize_to_tensor %82[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %117 = loom.sync ins(%116 : tensor<64x32xf16>) outs(%81 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %82 : memref<64x32xf16>
                    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %117, %114 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%81 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %122 = arith.mulf %in_7, %in_8 : f16
                      %123 = arith.addf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %108 : memref<f16>
                    loom.semaphore_give %69 : memref<64x32xf16>
                    %119 = loom.sync ins(%118 : tensor<64x32xf16>) outs(%79 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %80 : memref<64x32xf16>
                    %120 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %46, %22, %55)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%120], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %121 = loom.bufferize_to_memref %119 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %121, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %78 : memref<64x32xf16>
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
                    %27 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %28 = loom.init_tensor %27[64] : memref<64xf16> -> tensor<64xf16>
                    %29 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %30 = arith.muli %arg12, %c2 : index
                    %31 = arith.addi %arg9, %30 : index
                    %32 = arith.muli %arg11, %c2 : index
                    %33 = arith.muli %arg8, %c4 : index
                    %34 = arith.addi %32, %33 : index
                    %35 = arith.addi %32, %c1 : index
                    %36 = arith.addi %35, %33 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %37 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %38 = loom.sync ins(%37 : tensor<64xf16>) outs(%28 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %26 : memref<64xf16>
                    %39 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %40 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
                    %41 = loom.init_tensor %40[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %42 = loom.semaphore_take %39 : memref<64x32xf16> -> memref<64x32xf16>
                    %43 = loom.init_tensor %42[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %44 = loom.broadcast ins(%38 : tensor<64xf16>) outs(%43 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %45 = arith.muli %arg12, %c1024 : index
                    %46 = arith.addi %24, %45 : index
                    %47 = arith.divui %22, %c64 : index
                    %48 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %49 = loom.semaphore_take %48 : memref<64x64xf16> -> memref<64x64xf16>
                    %50 = loom.init_tensor %49[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %51 = loom.semaphore_take %48 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %52 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %46, %47, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%31, %34], LR : [%31, %36]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %53 = loom.bufferize_to_tensor %51[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %54 = loom.sync ins(%53 : tensor<64x64xf16>) outs(%50 : tensor<64x64xf16>) -> tensor<64x64xf16>
                    loom.semaphore_give %51 : memref<64x64xf16>
                    %55 = arith.muli %arg10, %c32 : index
                    %56 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %57 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                    %58 = loom.init_tensor %57[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %59 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %60 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %55)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%60], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %61 = arith.addi %30, %c1 : index
                    %62 = arith.addi %arg10, %32 : index
                    %63 = arith.addi %62, %33 : index
                    loom.copy %reinterpret_cast_3, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%30, %63], LR : [%61, %63]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %64 = loom.bufferize_to_tensor %59[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %65 = loom.sync ins(%64 : tensor<64x32xf16>) outs(%58 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %59 : memref<64x32xf16>
                    %66 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %67 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %68 = loom.init_tensor %67[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %69 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %70 = loom.init_tensor %69[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %71 = linalg.fill ins(%cst : f16) outs(%68 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %72 = linalg.matmul ins(%54, %65 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%71 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %57 : memref<64x32xf16>
                    loom.semaphore_give %49 : memref<64x64xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %44 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%70 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %122 = math.exp %in_7 : f16
                      %123 = arith.mulf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %67 : memref<64x32xf16>
                    loom.semaphore_give %42 : memref<64x32xf16>
                    %74 = arith.addi %21, %c1 : index
                    %75 = arith.muli %74, %c64 : index
                    %76 = arith.ceildivui %75, %c64 : index
                    %77 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %78 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %79 = loom.init_tensor %78[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %80 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %81 = loom.init_tensor %80[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %82 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %83 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %84 = loom.init_tensor %83[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %85 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %86 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %87 = loom.init_tensor %86[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %88 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %89 = loom.alloc [64] on @L1 : memref<64xf16>
                    %90 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %91 = loom.init_tensor %90[64] : memref<64xf16> -> tensor<64xf16>
                    %92 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %93 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %94 = loom.init_tensor %93[64] : memref<64xf16> -> tensor<64xf16>
                    %95 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %96 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %97 = loom.semaphore_take %96 : memref<32x64xf16> -> memref<32x64xf16>
                    %98 = loom.init_tensor %97[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %99 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %100 = loom.semaphore_take %99 : memref<32x64xf16> -> memref<32x64xf16>
                    %101 = loom.init_tensor %100[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %102 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %103 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %104 = loom.init_tensor %103[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %105 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %106 = scf.for %arg15 = %c0 to %76 step %c1 iter_args(%arg16 = %73) -> (tensor<64x32xf16>) {
                      %122 = arith.muli %arg15, %c64 : index
                      %123 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %47, %24, %122)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%123], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %124 = loom.bufferize_to_tensor %88[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %125 = loom.sync ins(%124 : tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) -> tensor<64x64xf16>
                      loom.semaphore_give %88 : memref<64x64xf16>
                      %126 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%126], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %95 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %127 = loom.bufferize_to_tensor %95[64] : memref<64xf16> -> tensor<64xf16>
                      %128 = loom.sync ins(%127 : tensor<64xf16>) outs(%94 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %95 : memref<64xf16>
                      %129 = loom.broadcast ins(%38 : tensor<64xf16>) outs(%41 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %130 = loom.broadcast ins(%128 : tensor<64xf16>) outs(%98 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %93 : memref<64xf16>
                      %131 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%131], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %132 = loom.bufferize_to_tensor %92[64] : memref<64xf16> -> tensor<64xf16>
                      %133 = loom.sync ins(%132 : tensor<64xf16>) outs(%91 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %92 : memref<64xf16>
                      %134 = loom.broadcast ins(%133 : tensor<64xf16>) outs(%101 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %90 : memref<64xf16>
                      %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%125, %129, %130, %134 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %143 = arith.subf %in_11, %in_12 : f16
                        %144 = math.exp %143 : f16
                        %145 = arith.mulf %in, %144 : f16
                        %146 = arith.mulf %145, %in_13 : f16
                        linalg.yield %146 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %100 : memref<32x64xf16>
                      loom.semaphore_give %97 : memref<32x64xf16>
                      loom.semaphore_give %40 : memref<64x32xf16>
                      %136 = arith.addi %122, %45 : index
                      %137 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %136, %22, %55)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%137], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %138 = loom.bufferize_to_tensor %105[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %139 = loom.sync ins(%138 : tensor<64x32xf16>) outs(%104 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %105 : memref<64x32xf16>
                      %140 = linalg.fill ins(%cst : f16) outs(%84 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %141 = linalg.matmul ins(%135, %139 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%140 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %103 : memref<64x32xf16>
                      loom.semaphore_give %86 : memref<64x64xf16>
                      %142 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %141 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %143 = arith.addf %in, %in_11 : f16
                        linalg.yield %143 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %83 : memref<64x32xf16>
                      scf.yield %142 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %27 : memref<64xf16>
                    %107 = loom.alloc [1] on @L1 : memref<f16>
                    %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %109 = loom.init_tensor %108[] : memref<f16> -> tensor<f16>
                    %110 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %111 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%111], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %112 = arith.addi %33, %c3 : index
                    loom.copy %reinterpret_cast_4, %110 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %33], LR : [%c7, %112]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %113 = loom.bufferize_to_tensor %110[] : memref<f16> -> tensor<f16>
                    %114 = loom.sync ins(%113 : tensor<f16>) outs(%109 : tensor<f16>) -> tensor<f16>
                    loom.semaphore_give %110 : memref<f16>
                    %115 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %46, %22, %55)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%115], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %116 = loom.bufferize_to_tensor %82[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %117 = loom.sync ins(%116 : tensor<64x32xf16>) outs(%81 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %82 : memref<64x32xf16>
                    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %117, %114 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%81 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %122 = arith.mulf %in_7, %in_8 : f16
                      %123 = arith.addf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %108 : memref<f16>
                    loom.semaphore_give %69 : memref<64x32xf16>
                    %119 = loom.sync ins(%118 : tensor<64x32xf16>) outs(%79 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %80 : memref<64x32xf16>
                    %120 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %46, %22, %55)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%120], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %121 = loom.bufferize_to_memref %119 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %121, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %63], LR : [%31, %63]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %78 : memref<64x32xf16>
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
                  %26 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %27 = loom.init_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                  %28 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %29 = arith.muli %arg11, %c4 : index
                  %30 = arith.addi %arg9, %29 : index
                  %31 = arith.muli %arg12, %c2 : index
                  %32 = arith.muli %arg8, %c4 : index
                  %33 = arith.addi %31, %32 : index
                  %34 = arith.addi %31, %c1 : index
                  %35 = arith.addi %34, %32 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %33], LR : [%30, %35]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %36 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %37 = loom.sync ins(%36 : tensor<64xf16>) outs(%27 : tensor<64xf16>) -> tensor<64xf16>
                  loom.semaphore_give %25 : memref<64xf16>
                  %38 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %39 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                  %40 = loom.init_tensor %39[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %41 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                  %42 = loom.init_tensor %41[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %43 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%42 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                  %44 = arith.muli %arg12, %c1024 : index
                  %45 = arith.addi %23, %44 : index
                  %46 = arith.divui %21, %c64 : index
                  %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                  %49 = loom.init_tensor %48[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %50 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_0 = arith.constant 0 : index
                  %51 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %45, %46, %c0_0)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  loom.copy %reinterpret_cast_1, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %33], LR : [%30, %35]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %52 = loom.bufferize_to_tensor %50[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %53 = loom.sync ins(%52 : tensor<64x64xf16>) outs(%49 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  loom.semaphore_give %50 : memref<64x64xf16>
                  %54 = arith.muli %arg10, %c32 : index
                  %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                  %57 = loom.init_tensor %56[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %58 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                  %c0_2 = arith.constant 0 : index
                  %59 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %c0_2, %54)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                  %60 = arith.addi %29, %c3 : index
                  %61 = arith.addi %arg10, %31 : index
                  %62 = arith.addi %61, %32 : index
                  loom.copy %reinterpret_cast_3, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%29, %62], LR : [%60, %62]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                  %63 = loom.bufferize_to_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %64 = loom.sync ins(%63 : tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %58 : memref<64x32xf16>
                  %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
                  %67 = loom.init_tensor %66[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %68 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
                  %69 = loom.init_tensor %68[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %70 = linalg.fill ins(%cst : f16) outs(%67 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %71 = linalg.matmul ins(%53, %64 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%70 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %56 : memref<64x32xf16>
                  loom.semaphore_give %48 : memref<64x64xf16>
                  %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %43 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%69 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_7: f16, %out: f16):
                    %121 = math.exp %in_7 : f16
                    %122 = arith.mulf %in, %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %66 : memref<64x32xf16>
                  loom.semaphore_give %41 : memref<64x32xf16>
                  %73 = arith.addi %arg9, %c1 : index
                  %74 = arith.muli %73, %c64 : index
                  %75 = arith.ceildivui %74, %c64 : index
                  %76 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %77 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
                  %78 = loom.init_tensor %77[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %79 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
                  %80 = loom.init_tensor %79[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %81 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
                  %82 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
                  %83 = loom.init_tensor %82[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %84 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %85 = loom.semaphore_take %84 : memref<64x64xf16> -> memref<64x64xf16>
                  %86 = loom.init_tensor %85[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %87 = loom.semaphore_take %84 : memref<64x64xf16> -> memref<64x64xf16>
                  %88 = loom.alloc [64] on @L1 : memref<64xf16>
                  %89 = loom.semaphore_take %88 : memref<64xf16> -> memref<64xf16>
                  %90 = loom.init_tensor %89[64] : memref<64xf16> -> tensor<64xf16>
                  %91 = loom.semaphore_take %88 : memref<64xf16> -> memref<64xf16>
                  %92 = loom.semaphore_take %88 : memref<64xf16> -> memref<64xf16>
                  %93 = loom.init_tensor %92[64] : memref<64xf16> -> tensor<64xf16>
                  %94 = loom.semaphore_take %88 : memref<64xf16> -> memref<64xf16>
                  %95 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %96 = loom.semaphore_take %95 : memref<32x64xf16> -> memref<32x64xf16>
                  %97 = loom.init_tensor %96[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %98 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %99 = loom.semaphore_take %98 : memref<32x64xf16> -> memref<32x64xf16>
                  %100 = loom.init_tensor %99[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %101 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %102 = loom.semaphore_take %101 : memref<64x32xf16> -> memref<64x32xf16>
                  %103 = loom.init_tensor %102[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %104 = loom.semaphore_take %101 : memref<64x32xf16> -> memref<64x32xf16>
                  %105 = scf.for %arg14 = %c0 to %75 step %c1 iter_args(%arg15 = %72) -> (tensor<64x32xf16>) {
                    %121 = arith.muli %arg14, %c64 : index
                    %122 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %46, %23, %121)
                    %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%122], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_7, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %123 = loom.bufferize_to_tensor %87[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %124 = loom.sync ins(%123 : tensor<64x64xf16>) outs(%86 : tensor<64x64xf16>) -> tensor<64x64xf16>
                    loom.semaphore_give %87 : memref<64x64xf16>
                    %125 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %121)
                    %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%125], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_8, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %126 = loom.bufferize_to_tensor %94[64] : memref<64xf16> -> tensor<64xf16>
                    %127 = loom.sync ins(%126 : tensor<64xf16>) outs(%93 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %94 : memref<64xf16>
                    %128 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%40 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                    %129 = loom.broadcast ins(%127 : tensor<64xf16>) outs(%97 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                    loom.semaphore_give %92 : memref<64xf16>
                    %130 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %121)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%130], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %131 = loom.bufferize_to_tensor %91[64] : memref<64xf16> -> tensor<64xf16>
                    %132 = loom.sync ins(%131 : tensor<64xf16>) outs(%90 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %91 : memref<64xf16>
                    %133 = loom.broadcast ins(%132 : tensor<64xf16>) outs(%100 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                    loom.semaphore_give %89 : memref<64xf16>
                    %134 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %128, %129, %133 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%86 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                      %142 = arith.subf %in_11, %in_12 : f16
                      %143 = math.exp %142 : f16
                      %144 = arith.mulf %in, %143 : f16
                      %145 = arith.mulf %144, %in_13 : f16
                      linalg.yield %145 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %99 : memref<32x64xf16>
                    loom.semaphore_give %96 : memref<32x64xf16>
                    loom.semaphore_give %39 : memref<64x32xf16>
                    %135 = arith.addi %121, %44 : index
                    %136 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %135, %21, %54)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%136], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %137 = loom.bufferize_to_tensor %104[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %138 = loom.sync ins(%137 : tensor<64x32xf16>) outs(%103 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %104 : memref<64x32xf16>
                    %139 = linalg.fill ins(%cst : f16) outs(%83 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %140 = linalg.matmul ins(%134, %138 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%139 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %102 : memref<64x32xf16>
                    loom.semaphore_give %85 : memref<64x64xf16>
                    %141 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %140 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_11: f16, %out: f16):
                      %142 = arith.addf %in, %in_11 : f16
                      linalg.yield %142 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %82 : memref<64x32xf16>
                    scf.yield %141 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %26 : memref<64xf16>
                  %106 = loom.alloc [1] on @L1 : memref<f16>
                  %107 = loom.semaphore_take %106 : memref<f16> -> memref<f16>
                  %108 = loom.init_tensor %107[] : memref<f16> -> tensor<f16>
                  %109 = loom.semaphore_take %106 : memref<f16> -> memref<f16>
                  %110 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%110], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %111 = arith.addi %32, %c3 : index
                  loom.copy %reinterpret_cast_4, %109 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %111]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %112 = loom.bufferize_to_tensor %109[] : memref<f16> -> tensor<f16>
                  %113 = loom.sync ins(%112 : tensor<f16>) outs(%108 : tensor<f16>) -> tensor<f16>
                  loom.semaphore_give %109 : memref<f16>
                  %114 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %21, %54)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%114], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_5, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %115 = loom.bufferize_to_tensor %81[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %116 = loom.sync ins(%115 : tensor<64x32xf16>) outs(%80 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %81 : memref<64x32xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%105, %116, %113 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%80 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                    %121 = arith.mulf %in_7, %in_8 : f16
                    %122 = arith.addf %in, %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %107 : memref<f16>
                  loom.semaphore_give %68 : memref<64x32xf16>
                  %118 = loom.sync ins(%117 : tensor<64x32xf16>) outs(%78 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %79 : memref<64x32xf16>
                  %119 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %21, %54)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%119], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %120, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %77 : memref<64x32xf16>
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
                  %26 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %27 = loom.init_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                  %28 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%28], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                  %29 = arith.muli %arg12, %c4 : index
                  %30 = arith.addi %arg9, %29 : index
                  %31 = arith.muli %arg11, %c2 : index
                  %32 = arith.muli %arg8, %c4 : index
                  %33 = arith.addi %31, %32 : index
                  %34 = arith.addi %31, %c1 : index
                  %35 = arith.addi %34, %32 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %33], LR : [%30, %35]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %36 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %37 = loom.sync ins(%36 : tensor<64xf16>) outs(%27 : tensor<64xf16>) -> tensor<64xf16>
                  loom.semaphore_give %25 : memref<64xf16>
                  %38 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %39 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                  %40 = loom.init_tensor %39[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %41 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                  %42 = loom.init_tensor %41[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %43 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%42 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                  %44 = arith.muli %arg12, %c1024 : index
                  %45 = arith.addi %23, %44 : index
                  %46 = arith.divui %21, %c64 : index
                  %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                  %49 = loom.init_tensor %48[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %50 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                  %c0_0 = arith.constant 0 : index
                  %51 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %45, %46, %c0_0)
                  %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                  loom.copy %reinterpret_cast_1, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %33], LR : [%30, %35]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                  %52 = loom.bufferize_to_tensor %50[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %53 = loom.sync ins(%52 : tensor<64x64xf16>) outs(%49 : tensor<64x64xf16>) -> tensor<64x64xf16>
                  loom.semaphore_give %50 : memref<64x64xf16>
                  %54 = arith.muli %arg10, %c32 : index
                  %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                  %57 = loom.init_tensor %56[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %58 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                  %c0_2 = arith.constant 0 : index
                  %59 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %22, %21, %c0_2, %54)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                  %60 = arith.addi %29, %c3 : index
                  %61 = arith.addi %arg10, %31 : index
                  %62 = arith.addi %61, %32 : index
                  loom.copy %reinterpret_cast_3, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%29, %62], LR : [%60, %62]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                  %63 = loom.bufferize_to_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %64 = loom.sync ins(%63 : tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %58 : memref<64x32xf16>
                  %65 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %66 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
                  %67 = loom.init_tensor %66[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %68 = loom.semaphore_take %65 : memref<64x32xf16> -> memref<64x32xf16>
                  %69 = loom.init_tensor %68[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %70 = linalg.fill ins(%cst : f16) outs(%67 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  %71 = linalg.matmul ins(%53, %64 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%70 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %56 : memref<64x32xf16>
                  loom.semaphore_give %48 : memref<64x64xf16>
                  %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %43 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%69 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_7: f16, %out: f16):
                    %121 = math.exp %in_7 : f16
                    %122 = arith.mulf %in, %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %66 : memref<64x32xf16>
                  loom.semaphore_give %41 : memref<64x32xf16>
                  %73 = arith.addi %arg9, %c1 : index
                  %74 = arith.muli %73, %c64 : index
                  %75 = arith.ceildivui %74, %c64 : index
                  %76 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %77 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
                  %78 = loom.init_tensor %77[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %79 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
                  %80 = loom.init_tensor %79[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %81 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
                  %82 = loom.semaphore_take %76 : memref<64x32xf16> -> memref<64x32xf16>
                  %83 = loom.init_tensor %82[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %84 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %85 = loom.semaphore_take %84 : memref<64x64xf16> -> memref<64x64xf16>
                  %86 = loom.init_tensor %85[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %87 = loom.semaphore_take %84 : memref<64x64xf16> -> memref<64x64xf16>
                  %88 = loom.alloc [64] on @L1 : memref<64xf16>
                  %89 = loom.semaphore_take %88 : memref<64xf16> -> memref<64xf16>
                  %90 = loom.init_tensor %89[64] : memref<64xf16> -> tensor<64xf16>
                  %91 = loom.semaphore_take %88 : memref<64xf16> -> memref<64xf16>
                  %92 = loom.semaphore_take %88 : memref<64xf16> -> memref<64xf16>
                  %93 = loom.init_tensor %92[64] : memref<64xf16> -> tensor<64xf16>
                  %94 = loom.semaphore_take %88 : memref<64xf16> -> memref<64xf16>
                  %95 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %96 = loom.semaphore_take %95 : memref<32x64xf16> -> memref<32x64xf16>
                  %97 = loom.init_tensor %96[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %98 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                  %99 = loom.semaphore_take %98 : memref<32x64xf16> -> memref<32x64xf16>
                  %100 = loom.init_tensor %99[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                  %101 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %102 = loom.semaphore_take %101 : memref<64x32xf16> -> memref<64x32xf16>
                  %103 = loom.init_tensor %102[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %104 = loom.semaphore_take %101 : memref<64x32xf16> -> memref<64x32xf16>
                  %105 = scf.for %arg14 = %c0 to %75 step %c1 iter_args(%arg15 = %72) -> (tensor<64x32xf16>) {
                    %121 = arith.muli %arg14, %c64 : index
                    %122 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %46, %23, %121)
                    %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%122], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_7, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %123 = loom.bufferize_to_tensor %87[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %124 = loom.sync ins(%123 : tensor<64x64xf16>) outs(%86 : tensor<64x64xf16>) -> tensor<64x64xf16>
                    loom.semaphore_give %87 : memref<64x64xf16>
                    %125 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %121)
                    %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%125], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_8, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %126 = loom.bufferize_to_tensor %94[64] : memref<64xf16> -> tensor<64xf16>
                    %127 = loom.sync ins(%126 : tensor<64xf16>) outs(%93 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %94 : memref<64xf16>
                    %128 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%40 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                    %129 = loom.broadcast ins(%127 : tensor<64xf16>) outs(%97 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                    loom.semaphore_give %92 : memref<64xf16>
                    %130 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %121)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%130], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %131 = loom.bufferize_to_tensor %91[64] : memref<64xf16> -> tensor<64xf16>
                    %132 = loom.sync ins(%131 : tensor<64xf16>) outs(%90 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %91 : memref<64xf16>
                    %133 = loom.broadcast ins(%132 : tensor<64xf16>) outs(%100 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                    loom.semaphore_give %89 : memref<64xf16>
                    %134 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%124, %128, %129, %133 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%86 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                      %142 = arith.subf %in_11, %in_12 : f16
                      %143 = math.exp %142 : f16
                      %144 = arith.mulf %in, %143 : f16
                      %145 = arith.mulf %144, %in_13 : f16
                      linalg.yield %145 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %99 : memref<32x64xf16>
                    loom.semaphore_give %96 : memref<32x64xf16>
                    loom.semaphore_give %39 : memref<64x32xf16>
                    %135 = arith.addi %121, %44 : index
                    %136 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %135, %21, %54)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%136], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %137 = loom.bufferize_to_tensor %104[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %138 = loom.sync ins(%137 : tensor<64x32xf16>) outs(%103 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %104 : memref<64x32xf16>
                    %139 = linalg.fill ins(%cst : f16) outs(%83 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %140 = linalg.matmul ins(%134, %138 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%139 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %102 : memref<64x32xf16>
                    loom.semaphore_give %85 : memref<64x64xf16>
                    %141 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg15, %140 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_11: f16, %out: f16):
                      %142 = arith.addf %in, %in_11 : f16
                      linalg.yield %142 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %82 : memref<64x32xf16>
                    scf.yield %141 : tensor<64x32xf16>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %26 : memref<64xf16>
                  %106 = loom.alloc [1] on @L1 : memref<f16>
                  %107 = loom.semaphore_take %106 : memref<f16> -> memref<f16>
                  %108 = loom.init_tensor %107[] : memref<f16> -> tensor<f16>
                  %109 = loom.semaphore_take %106 : memref<f16> -> memref<f16>
                  %110 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%110], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                  %111 = arith.addi %32, %c3 : index
                  loom.copy %reinterpret_cast_4, %109 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %111]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %112 = loom.bufferize_to_tensor %109[] : memref<f16> -> tensor<f16>
                  %113 = loom.sync ins(%112 : tensor<f16>) outs(%108 : tensor<f16>) -> tensor<f16>
                  loom.semaphore_give %109 : memref<f16>
                  %114 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %21, %54)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%114], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %reinterpret_cast_5, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                  %115 = loom.bufferize_to_tensor %81[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %116 = loom.sync ins(%115 : tensor<64x32xf16>) outs(%80 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %81 : memref<64x32xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%105, %116, %113 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%80 : tensor<64x32xf16>) {
                  ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                    %121 = arith.mulf %in_7, %in_8 : f16
                    %122 = arith.addf %in, %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %107 : memref<f16>
                  loom.semaphore_give %68 : memref<64x32xf16>
                  %118 = loom.sync ins(%117 : tensor<64x32xf16>) outs(%78 : tensor<64x32xf16>) -> tensor<64x32xf16>
                  loom.semaphore_give %79 : memref<64x32xf16>
                  %119 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %21, %54)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%119], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %120, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%30, %62], LR : [%30, %62]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                  loom.semaphore_give %77 : memref<64x32xf16>
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
                    %27 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %28 = loom.init_tensor %27[64] : memref<64xf16> -> tensor<64xf16>
                    %29 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %30 = arith.muli %arg11, %c2 : index
                    %31 = arith.addi %arg9, %30 : index
                    %32 = arith.muli %arg8, %c4 : index
                    %33 = arith.addi %31, %32 : index
                    %34 = arith.muli %arg12, %c2 : index
                    %35 = arith.addi %34, %c1 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %34], LR : [%33, %35]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %36 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %37 = loom.sync ins(%36 : tensor<64xf16>) outs(%28 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %26 : memref<64xf16>
                    %38 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %39 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                    %40 = loom.init_tensor %39[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %41 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                    %42 = loom.init_tensor %41[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %43 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%42 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %44 = arith.muli %arg12, %c1024 : index
                    %45 = arith.addi %24, %44 : index
                    %46 = arith.divui %22, %c64 : index
                    %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                    %49 = loom.init_tensor %48[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %50 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %51 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %45, %46, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %34], LR : [%33, %35]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %52 = loom.bufferize_to_tensor %50[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %53 = loom.sync ins(%52 : tensor<64x64xf16>) outs(%49 : tensor<64x64xf16>) -> tensor<64x64xf16>
                    loom.semaphore_give %50 : memref<64x64xf16>
                    %54 = arith.muli %arg10, %c32 : index
                    %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                    %57 = loom.init_tensor %56[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %58 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %59 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %54)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %60 = arith.addi %30, %32 : index
                    %61 = arith.addi %30, %c1 : index
                    %62 = arith.addi %61, %32 : index
                    %63 = arith.addi %arg10, %34 : index
                    loom.copy %reinterpret_cast_3, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%60, %63], LR : [%62, %63]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %64 = loom.bufferize_to_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %65 = loom.sync ins(%64 : tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %58 : memref<64x32xf16>
                    %66 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %67 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %68 = loom.init_tensor %67[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %69 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %70 = loom.init_tensor %69[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %71 = linalg.fill ins(%cst : f16) outs(%68 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %72 = linalg.matmul ins(%53, %65 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%71 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %56 : memref<64x32xf16>
                    loom.semaphore_give %48 : memref<64x64xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %43 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%70 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %122 = math.exp %in_7 : f16
                      %123 = arith.mulf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %67 : memref<64x32xf16>
                    loom.semaphore_give %41 : memref<64x32xf16>
                    %74 = arith.addi %21, %c1 : index
                    %75 = arith.muli %74, %c64 : index
                    %76 = arith.ceildivui %75, %c64 : index
                    %77 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %78 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %79 = loom.init_tensor %78[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %80 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %81 = loom.init_tensor %80[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %82 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %83 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %84 = loom.init_tensor %83[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %85 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %86 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %87 = loom.init_tensor %86[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %88 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %89 = loom.alloc [64] on @L1 : memref<64xf16>
                    %90 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %91 = loom.init_tensor %90[64] : memref<64xf16> -> tensor<64xf16>
                    %92 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %93 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %94 = loom.init_tensor %93[64] : memref<64xf16> -> tensor<64xf16>
                    %95 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %96 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %97 = loom.semaphore_take %96 : memref<32x64xf16> -> memref<32x64xf16>
                    %98 = loom.init_tensor %97[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %99 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %100 = loom.semaphore_take %99 : memref<32x64xf16> -> memref<32x64xf16>
                    %101 = loom.init_tensor %100[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %102 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %103 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %104 = loom.init_tensor %103[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %105 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %106 = scf.for %arg15 = %c0 to %76 step %c1 iter_args(%arg16 = %73) -> (tensor<64x32xf16>) {
                      %122 = arith.muli %arg15, %c64 : index
                      %123 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %46, %24, %122)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%123], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %124 = loom.bufferize_to_tensor %88[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %125 = loom.sync ins(%124 : tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) -> tensor<64x64xf16>
                      loom.semaphore_give %88 : memref<64x64xf16>
                      %126 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%126], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %95 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %127 = loom.bufferize_to_tensor %95[64] : memref<64xf16> -> tensor<64xf16>
                      %128 = loom.sync ins(%127 : tensor<64xf16>) outs(%94 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %95 : memref<64xf16>
                      %129 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%40 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %130 = loom.broadcast ins(%128 : tensor<64xf16>) outs(%98 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %93 : memref<64xf16>
                      %131 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%131], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %132 = loom.bufferize_to_tensor %92[64] : memref<64xf16> -> tensor<64xf16>
                      %133 = loom.sync ins(%132 : tensor<64xf16>) outs(%91 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %92 : memref<64xf16>
                      %134 = loom.broadcast ins(%133 : tensor<64xf16>) outs(%101 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %90 : memref<64xf16>
                      %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%125, %129, %130, %134 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %143 = arith.subf %in_11, %in_12 : f16
                        %144 = math.exp %143 : f16
                        %145 = arith.mulf %in, %144 : f16
                        %146 = arith.mulf %145, %in_13 : f16
                        linalg.yield %146 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %100 : memref<32x64xf16>
                      loom.semaphore_give %97 : memref<32x64xf16>
                      loom.semaphore_give %39 : memref<64x32xf16>
                      %136 = arith.addi %122, %44 : index
                      %137 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %136, %22, %54)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%137], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %138 = loom.bufferize_to_tensor %105[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %139 = loom.sync ins(%138 : tensor<64x32xf16>) outs(%104 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %105 : memref<64x32xf16>
                      %140 = linalg.fill ins(%cst : f16) outs(%84 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %141 = linalg.matmul ins(%135, %139 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%140 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %103 : memref<64x32xf16>
                      loom.semaphore_give %86 : memref<64x64xf16>
                      %142 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %141 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %143 = arith.addf %in, %in_11 : f16
                        linalg.yield %143 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %83 : memref<64x32xf16>
                      scf.yield %142 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %27 : memref<64xf16>
                    %107 = loom.alloc [1] on @L1 : memref<f16>
                    %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %109 = loom.init_tensor %108[] : memref<f16> -> tensor<f16>
                    %110 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %111 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%111], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %112 = arith.addi %32, %c3 : index
                    loom.copy %reinterpret_cast_4, %110 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%32, %c0], LR : [%112, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %113 = loom.bufferize_to_tensor %110[] : memref<f16> -> tensor<f16>
                    %114 = loom.sync ins(%113 : tensor<f16>) outs(%109 : tensor<f16>) -> tensor<f16>
                    loom.semaphore_give %110 : memref<f16>
                    %115 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %22, %54)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%115], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %116 = loom.bufferize_to_tensor %82[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %117 = loom.sync ins(%116 : tensor<64x32xf16>) outs(%81 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %82 : memref<64x32xf16>
                    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %117, %114 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%81 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %122 = arith.mulf %in_7, %in_8 : f16
                      %123 = arith.addf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %108 : memref<f16>
                    loom.semaphore_give %69 : memref<64x32xf16>
                    %119 = loom.sync ins(%118 : tensor<64x32xf16>) outs(%79 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %80 : memref<64x32xf16>
                    %120 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %22, %54)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%120], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %121 = loom.bufferize_to_memref %119 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %121, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %78 : memref<64x32xf16>
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
                    %27 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %28 = loom.init_tensor %27[64] : memref<64xf16> -> tensor<64xf16>
                    %29 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %30 = arith.muli %arg12, %c2 : index
                    %31 = arith.addi %arg9, %30 : index
                    %32 = arith.muli %arg8, %c4 : index
                    %33 = arith.addi %31, %32 : index
                    %34 = arith.muli %arg11, %c2 : index
                    %35 = arith.addi %34, %c1 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %34], LR : [%33, %35]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %36 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %37 = loom.sync ins(%36 : tensor<64xf16>) outs(%28 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %26 : memref<64xf16>
                    %38 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %39 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                    %40 = loom.init_tensor %39[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %41 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                    %42 = loom.init_tensor %41[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %43 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%42 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %44 = arith.muli %arg12, %c1024 : index
                    %45 = arith.addi %24, %44 : index
                    %46 = arith.divui %22, %c64 : index
                    %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                    %49 = loom.init_tensor %48[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %50 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %51 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %45, %46, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%33, %34], LR : [%33, %35]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %52 = loom.bufferize_to_tensor %50[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %53 = loom.sync ins(%52 : tensor<64x64xf16>) outs(%49 : tensor<64x64xf16>) -> tensor<64x64xf16>
                    loom.semaphore_give %50 : memref<64x64xf16>
                    %54 = arith.muli %arg10, %c32 : index
                    %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                    %57 = loom.init_tensor %56[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %58 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %59 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %54)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %60 = arith.addi %30, %32 : index
                    %61 = arith.addi %30, %c1 : index
                    %62 = arith.addi %61, %32 : index
                    %63 = arith.addi %arg10, %34 : index
                    loom.copy %reinterpret_cast_3, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%60, %63], LR : [%62, %63]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %64 = loom.bufferize_to_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %65 = loom.sync ins(%64 : tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %58 : memref<64x32xf16>
                    %66 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %67 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %68 = loom.init_tensor %67[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %69 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %70 = loom.init_tensor %69[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %71 = linalg.fill ins(%cst : f16) outs(%68 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %72 = linalg.matmul ins(%53, %65 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%71 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %56 : memref<64x32xf16>
                    loom.semaphore_give %48 : memref<64x64xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %43 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%70 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %122 = math.exp %in_7 : f16
                      %123 = arith.mulf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %67 : memref<64x32xf16>
                    loom.semaphore_give %41 : memref<64x32xf16>
                    %74 = arith.addi %21, %c1 : index
                    %75 = arith.muli %74, %c64 : index
                    %76 = arith.ceildivui %75, %c64 : index
                    %77 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %78 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %79 = loom.init_tensor %78[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %80 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %81 = loom.init_tensor %80[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %82 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %83 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %84 = loom.init_tensor %83[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %85 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %86 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %87 = loom.init_tensor %86[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %88 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %89 = loom.alloc [64] on @L1 : memref<64xf16>
                    %90 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %91 = loom.init_tensor %90[64] : memref<64xf16> -> tensor<64xf16>
                    %92 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %93 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %94 = loom.init_tensor %93[64] : memref<64xf16> -> tensor<64xf16>
                    %95 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %96 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %97 = loom.semaphore_take %96 : memref<32x64xf16> -> memref<32x64xf16>
                    %98 = loom.init_tensor %97[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %99 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %100 = loom.semaphore_take %99 : memref<32x64xf16> -> memref<32x64xf16>
                    %101 = loom.init_tensor %100[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %102 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %103 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %104 = loom.init_tensor %103[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %105 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %106 = scf.for %arg15 = %c0 to %76 step %c1 iter_args(%arg16 = %73) -> (tensor<64x32xf16>) {
                      %122 = arith.muli %arg15, %c64 : index
                      %123 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %46, %24, %122)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%123], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %124 = loom.bufferize_to_tensor %88[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %125 = loom.sync ins(%124 : tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) -> tensor<64x64xf16>
                      loom.semaphore_give %88 : memref<64x64xf16>
                      %126 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%126], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %95 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %127 = loom.bufferize_to_tensor %95[64] : memref<64xf16> -> tensor<64xf16>
                      %128 = loom.sync ins(%127 : tensor<64xf16>) outs(%94 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %95 : memref<64xf16>
                      %129 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%40 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %130 = loom.broadcast ins(%128 : tensor<64xf16>) outs(%98 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %93 : memref<64xf16>
                      %131 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%131], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %132 = loom.bufferize_to_tensor %92[64] : memref<64xf16> -> tensor<64xf16>
                      %133 = loom.sync ins(%132 : tensor<64xf16>) outs(%91 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %92 : memref<64xf16>
                      %134 = loom.broadcast ins(%133 : tensor<64xf16>) outs(%101 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %90 : memref<64xf16>
                      %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%125, %129, %130, %134 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %143 = arith.subf %in_11, %in_12 : f16
                        %144 = math.exp %143 : f16
                        %145 = arith.mulf %in, %144 : f16
                        %146 = arith.mulf %145, %in_13 : f16
                        linalg.yield %146 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %100 : memref<32x64xf16>
                      loom.semaphore_give %97 : memref<32x64xf16>
                      loom.semaphore_give %39 : memref<64x32xf16>
                      %136 = arith.addi %122, %44 : index
                      %137 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %136, %22, %54)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%137], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %138 = loom.bufferize_to_tensor %105[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %139 = loom.sync ins(%138 : tensor<64x32xf16>) outs(%104 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %105 : memref<64x32xf16>
                      %140 = linalg.fill ins(%cst : f16) outs(%84 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %141 = linalg.matmul ins(%135, %139 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%140 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %103 : memref<64x32xf16>
                      loom.semaphore_give %86 : memref<64x64xf16>
                      %142 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %141 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %143 = arith.addf %in, %in_11 : f16
                        linalg.yield %143 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %83 : memref<64x32xf16>
                      scf.yield %142 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %27 : memref<64xf16>
                    %107 = loom.alloc [1] on @L1 : memref<f16>
                    %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %109 = loom.init_tensor %108[] : memref<f16> -> tensor<f16>
                    %110 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %111 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%111], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %112 = arith.addi %32, %c3 : index
                    loom.copy %reinterpret_cast_4, %110 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%32, %c0], LR : [%112, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %113 = loom.bufferize_to_tensor %110[] : memref<f16> -> tensor<f16>
                    %114 = loom.sync ins(%113 : tensor<f16>) outs(%109 : tensor<f16>) -> tensor<f16>
                    loom.semaphore_give %110 : memref<f16>
                    %115 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %22, %54)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%115], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %116 = loom.bufferize_to_tensor %82[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %117 = loom.sync ins(%116 : tensor<64x32xf16>) outs(%81 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %82 : memref<64x32xf16>
                    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %117, %114 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%81 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %122 = arith.mulf %in_7, %in_8 : f16
                      %123 = arith.addf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %108 : memref<f16>
                    loom.semaphore_give %69 : memref<64x32xf16>
                    %119 = loom.sync ins(%118 : tensor<64x32xf16>) outs(%79 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %80 : memref<64x32xf16>
                    %120 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %22, %54)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%120], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %121 = loom.bufferize_to_memref %119 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %121, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %78 : memref<64x32xf16>
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
                    %27 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %28 = loom.init_tensor %27[64] : memref<64xf16> -> tensor<64xf16>
                    %29 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %30 = arith.muli %arg11, %c2 : index
                    %31 = arith.addi %arg9, %30 : index
                    %32 = arith.muli %arg8, %c4 : index
                    %33 = arith.addi %31, %32 : index
                    %34 = arith.muli %arg12, %c4 : index
                    %35 = arith.addi %34, %c3 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%33, %34], LR : [%33, %35]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %36 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %37 = loom.sync ins(%36 : tensor<64xf16>) outs(%28 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %26 : memref<64xf16>
                    %38 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %39 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                    %40 = loom.init_tensor %39[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %41 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                    %42 = loom.init_tensor %41[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %43 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%42 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %44 = arith.muli %arg12, %c1024 : index
                    %45 = arith.addi %24, %44 : index
                    %46 = arith.divui %22, %c64 : index
                    %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                    %49 = loom.init_tensor %48[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %50 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %51 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %45, %46, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%33, %34], LR : [%33, %35]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %52 = loom.bufferize_to_tensor %50[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %53 = loom.sync ins(%52 : tensor<64x64xf16>) outs(%49 : tensor<64x64xf16>) -> tensor<64x64xf16>
                    loom.semaphore_give %50 : memref<64x64xf16>
                    %54 = arith.muli %arg10, %c32 : index
                    %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                    %57 = loom.init_tensor %56[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %58 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %59 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %54)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %60 = arith.addi %30, %32 : index
                    %61 = arith.addi %30, %c1 : index
                    %62 = arith.addi %61, %32 : index
                    %63 = arith.addi %arg10, %34 : index
                    loom.copy %reinterpret_cast_3, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%60, %63], LR : [%62, %63]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %64 = loom.bufferize_to_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %65 = loom.sync ins(%64 : tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %58 : memref<64x32xf16>
                    %66 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %67 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %68 = loom.init_tensor %67[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %69 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %70 = loom.init_tensor %69[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %71 = linalg.fill ins(%cst : f16) outs(%68 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %72 = linalg.matmul ins(%53, %65 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%71 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %56 : memref<64x32xf16>
                    loom.semaphore_give %48 : memref<64x64xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %43 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%70 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %122 = math.exp %in_7 : f16
                      %123 = arith.mulf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %67 : memref<64x32xf16>
                    loom.semaphore_give %41 : memref<64x32xf16>
                    %74 = arith.addi %21, %c1 : index
                    %75 = arith.muli %74, %c64 : index
                    %76 = arith.ceildivui %75, %c64 : index
                    %77 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %78 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %79 = loom.init_tensor %78[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %80 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %81 = loom.init_tensor %80[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %82 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %83 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %84 = loom.init_tensor %83[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %85 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %86 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %87 = loom.init_tensor %86[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %88 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %89 = loom.alloc [64] on @L1 : memref<64xf16>
                    %90 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %91 = loom.init_tensor %90[64] : memref<64xf16> -> tensor<64xf16>
                    %92 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %93 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %94 = loom.init_tensor %93[64] : memref<64xf16> -> tensor<64xf16>
                    %95 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %96 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %97 = loom.semaphore_take %96 : memref<32x64xf16> -> memref<32x64xf16>
                    %98 = loom.init_tensor %97[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %99 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %100 = loom.semaphore_take %99 : memref<32x64xf16> -> memref<32x64xf16>
                    %101 = loom.init_tensor %100[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %102 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %103 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %104 = loom.init_tensor %103[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %105 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %106 = scf.for %arg15 = %c0 to %76 step %c1 iter_args(%arg16 = %73) -> (tensor<64x32xf16>) {
                      %122 = arith.muli %arg15, %c64 : index
                      %123 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %46, %24, %122)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%123], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %124 = loom.bufferize_to_tensor %88[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %125 = loom.sync ins(%124 : tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) -> tensor<64x64xf16>
                      loom.semaphore_give %88 : memref<64x64xf16>
                      %126 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%126], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %95 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %127 = loom.bufferize_to_tensor %95[64] : memref<64xf16> -> tensor<64xf16>
                      %128 = loom.sync ins(%127 : tensor<64xf16>) outs(%94 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %95 : memref<64xf16>
                      %129 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%40 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %130 = loom.broadcast ins(%128 : tensor<64xf16>) outs(%98 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %93 : memref<64xf16>
                      %131 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%131], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %132 = loom.bufferize_to_tensor %92[64] : memref<64xf16> -> tensor<64xf16>
                      %133 = loom.sync ins(%132 : tensor<64xf16>) outs(%91 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %92 : memref<64xf16>
                      %134 = loom.broadcast ins(%133 : tensor<64xf16>) outs(%101 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %90 : memref<64xf16>
                      %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%125, %129, %130, %134 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %143 = arith.subf %in_11, %in_12 : f16
                        %144 = math.exp %143 : f16
                        %145 = arith.mulf %in, %144 : f16
                        %146 = arith.mulf %145, %in_13 : f16
                        linalg.yield %146 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %100 : memref<32x64xf16>
                      loom.semaphore_give %97 : memref<32x64xf16>
                      loom.semaphore_give %39 : memref<64x32xf16>
                      %136 = arith.addi %122, %44 : index
                      %137 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %136, %22, %54)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%137], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %138 = loom.bufferize_to_tensor %105[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %139 = loom.sync ins(%138 : tensor<64x32xf16>) outs(%104 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %105 : memref<64x32xf16>
                      %140 = linalg.fill ins(%cst : f16) outs(%84 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %141 = linalg.matmul ins(%135, %139 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%140 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %103 : memref<64x32xf16>
                      loom.semaphore_give %86 : memref<64x64xf16>
                      %142 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %141 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %143 = arith.addf %in, %in_11 : f16
                        linalg.yield %143 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %83 : memref<64x32xf16>
                      scf.yield %142 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %27 : memref<64xf16>
                    %107 = loom.alloc [1] on @L1 : memref<f16>
                    %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %109 = loom.init_tensor %108[] : memref<f16> -> tensor<f16>
                    %110 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %111 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%111], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %112 = arith.addi %32, %c3 : index
                    loom.copy %reinterpret_cast_4, %110 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%32, %c0], LR : [%112, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %113 = loom.bufferize_to_tensor %110[] : memref<f16> -> tensor<f16>
                    %114 = loom.sync ins(%113 : tensor<f16>) outs(%109 : tensor<f16>) -> tensor<f16>
                    loom.semaphore_give %110 : memref<f16>
                    %115 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %22, %54)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%115], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %116 = loom.bufferize_to_tensor %82[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %117 = loom.sync ins(%116 : tensor<64x32xf16>) outs(%81 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %82 : memref<64x32xf16>
                    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %117, %114 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%81 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %122 = arith.mulf %in_7, %in_8 : f16
                      %123 = arith.addf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %108 : memref<f16>
                    loom.semaphore_give %69 : memref<64x32xf16>
                    %119 = loom.sync ins(%118 : tensor<64x32xf16>) outs(%79 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %80 : memref<64x32xf16>
                    %120 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %22, %54)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%120], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %121 = loom.bufferize_to_memref %119 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %121, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %78 : memref<64x32xf16>
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
                    %27 = loom.semaphore_take %25 : memref<64xf16> -> memref<64xf16>
                    %28 = loom.init_tensor %27[64] : memref<64xf16> -> tensor<64xf16>
                    %29 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %24)
                    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                    %30 = arith.muli %arg12, %c2 : index
                    %31 = arith.addi %arg9, %30 : index
                    %32 = arith.muli %arg8, %c4 : index
                    %33 = arith.addi %31, %32 : index
                    %34 = arith.muli %arg11, %c4 : index
                    %35 = arith.addi %34, %c3 : index
                    loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%33, %34], LR : [%33, %35]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %36 = loom.bufferize_to_tensor %26[64] : memref<64xf16> -> tensor<64xf16>
                    %37 = loom.sync ins(%36 : tensor<64xf16>) outs(%28 : tensor<64xf16>) -> tensor<64xf16>
                    loom.semaphore_give %26 : memref<64xf16>
                    %38 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %39 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                    %40 = loom.init_tensor %39[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %41 = loom.semaphore_take %38 : memref<64x32xf16> -> memref<64x32xf16>
                    %42 = loom.init_tensor %41[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %43 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%42 : tensor<64x32xf16>) dim(1) -> tensor<64x32xf16>
                    %44 = arith.muli %arg12, %c1024 : index
                    %45 = arith.addi %24, %44 : index
                    %46 = arith.divui %22, %c64 : index
                    %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                    %49 = loom.init_tensor %48[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %50 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
                    %c0_0 = arith.constant 0 : index
                    %51 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 64 + d2 * 64 + d3)>(%arg11, %45, %46, %c0_0)
                    %reinterpret_cast_1 = memref.reinterpret_cast %arg4 to offset: [%51], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
                    loom.copy %reinterpret_cast_1, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%33, %34], LR : [%33, %35]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
                    %52 = loom.bufferize_to_tensor %50[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %53 = loom.sync ins(%52 : tensor<64x64xf16>) outs(%49 : tensor<64x64xf16>) -> tensor<64x64xf16>
                    loom.semaphore_give %50 : memref<64x64xf16>
                    %54 = arith.muli %arg10, %c32 : index
                    %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                    %57 = loom.init_tensor %56[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %58 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
                    %c0_2 = arith.constant 0 : index
                    %59 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 2097152 + d1 * 262144 + d2 * 4096 + d3 * 64 + d4)>(%arg11, %23, %22, %c0_2, %54)
                    %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%59], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
                    %60 = arith.addi %30, %32 : index
                    %61 = arith.addi %30, %c1 : index
                    %62 = arith.addi %61, %32 : index
                    %63 = arith.addi %arg10, %34 : index
                    loom.copy %reinterpret_cast_3, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%60, %63], LR : [%62, %63]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
                    %64 = loom.bufferize_to_tensor %58[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %65 = loom.sync ins(%64 : tensor<64x32xf16>) outs(%57 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %58 : memref<64x32xf16>
                    %66 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %67 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %68 = loom.init_tensor %67[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %69 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
                    %70 = loom.init_tensor %69[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %71 = linalg.fill ins(%cst : f16) outs(%68 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    %72 = linalg.matmul ins(%53, %65 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%71 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %56 : memref<64x32xf16>
                    loom.semaphore_give %48 : memref<64x64xf16>
                    %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %43 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%70 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %out: f16):
                      %122 = math.exp %in_7 : f16
                      %123 = arith.mulf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %67 : memref<64x32xf16>
                    loom.semaphore_give %41 : memref<64x32xf16>
                    %74 = arith.addi %21, %c1 : index
                    %75 = arith.muli %74, %c64 : index
                    %76 = arith.ceildivui %75, %c64 : index
                    %77 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %78 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %79 = loom.init_tensor %78[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %80 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %81 = loom.init_tensor %80[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %82 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %83 = loom.semaphore_take %77 : memref<64x32xf16> -> memref<64x32xf16>
                    %84 = loom.init_tensor %83[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %85 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                    %86 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %87 = loom.init_tensor %86[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %88 = loom.semaphore_take %85 : memref<64x64xf16> -> memref<64x64xf16>
                    %89 = loom.alloc [64] on @L1 : memref<64xf16>
                    %90 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %91 = loom.init_tensor %90[64] : memref<64xf16> -> tensor<64xf16>
                    %92 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %93 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %94 = loom.init_tensor %93[64] : memref<64xf16> -> tensor<64xf16>
                    %95 = loom.semaphore_take %89 : memref<64xf16> -> memref<64xf16>
                    %96 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %97 = loom.semaphore_take %96 : memref<32x64xf16> -> memref<32x64xf16>
                    %98 = loom.init_tensor %97[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %99 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
                    %100 = loom.semaphore_take %99 : memref<32x64xf16> -> memref<32x64xf16>
                    %101 = loom.init_tensor %100[32, 64] : memref<32x64xf16> -> tensor<32x64xf16>
                    %102 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                    %103 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %104 = loom.init_tensor %103[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %105 = loom.semaphore_take %102 : memref<64x32xf16> -> memref<64x32xf16>
                    %106 = scf.for %arg15 = %c0 to %76 step %c1 iter_args(%arg16 = %73) -> (tensor<64x32xf16>) {
                      %122 = arith.muli %arg15, %c64 : index
                      %123 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %23, %46, %24, %122)
                      %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%123], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
                      loom.copy %reinterpret_cast_7, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                      %124 = loom.bufferize_to_tensor %88[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                      %125 = loom.sync ins(%124 : tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) -> tensor<64x64xf16>
                      loom.semaphore_give %88 : memref<64x64xf16>
                      %126 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%126], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_8, %95 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %127 = loom.bufferize_to_tensor %95[64] : memref<64xf16> -> tensor<64xf16>
                      %128 = loom.sync ins(%127 : tensor<64xf16>) outs(%94 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %95 : memref<64xf16>
                      %129 = loom.broadcast ins(%37 : tensor<64xf16>) outs(%40 : tensor<64x32xf16>) dim(1) -> tensor<64x64xf16>
                      %130 = loom.broadcast ins(%128 : tensor<64xf16>) outs(%98 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %93 : memref<64xf16>
                      %131 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %22, %23, %122)
                      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%131], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
                      loom.copy %reinterpret_cast_9, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
                      %132 = loom.bufferize_to_tensor %92[64] : memref<64xf16> -> tensor<64xf16>
                      %133 = loom.sync ins(%132 : tensor<64xf16>) outs(%91 : tensor<64xf16>) -> tensor<64xf16>
                      loom.semaphore_give %92 : memref<64xf16>
                      %134 = loom.broadcast ins(%133 : tensor<64xf16>) outs(%101 : tensor<32x64xf16>) dim(0) -> tensor<64x64xf16>
                      loom.semaphore_give %90 : memref<64xf16>
                      %135 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%125, %129, %130, %134 : tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>, tensor<64x64xf16>) outs(%87 : tensor<64x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
                        %143 = arith.subf %in_11, %in_12 : f16
                        %144 = math.exp %143 : f16
                        %145 = arith.mulf %in, %144 : f16
                        %146 = arith.mulf %145, %in_13 : f16
                        linalg.yield %146 : f16
                      } -> tensor<64x64xf16>
                      loom.semaphore_give %100 : memref<32x64xf16>
                      loom.semaphore_give %97 : memref<32x64xf16>
                      loom.semaphore_give %39 : memref<64x32xf16>
                      %136 = arith.addi %122, %44 : index
                      %137 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %136, %22, %54)
                      %reinterpret_cast_10 = memref.reinterpret_cast %arg3 to offset: [%137], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                      loom.copy %reinterpret_cast_10, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                      %138 = loom.bufferize_to_tensor %105[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                      %139 = loom.sync ins(%138 : tensor<64x32xf16>) outs(%104 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %105 : memref<64x32xf16>
                      %140 = linalg.fill ins(%cst : f16) outs(%84 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      %141 = linalg.matmul ins(%135, %139 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%140 : tensor<64x32xf16>) -> tensor<64x32xf16>
                      loom.semaphore_give %103 : memref<64x32xf16>
                      loom.semaphore_give %86 : memref<64x64xf16>
                      %142 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg16, %141 : tensor<64x32xf16>, tensor<64x32xf16>) outs(%arg16 : tensor<64x32xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %143 = arith.addf %in, %in_11 : f16
                        linalg.yield %143 : f16
                      } -> tensor<64x32xf16>
                      loom.semaphore_give %83 : memref<64x32xf16>
                      scf.yield %142 : tensor<64x32xf16>
                    } {loom.iter_type = #loom.iter_type<sequential>}
                    loom.semaphore_give %27 : memref<64xf16>
                    %107 = loom.alloc [1] on @L1 : memref<f16>
                    %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %109 = loom.init_tensor %108[] : memref<f16> -> tensor<f16>
                    %110 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                    %111 = affine.apply affine_map<(d0) -> (d0)>(%22)
                    %reinterpret_cast_4 = memref.reinterpret_cast %arg6 to offset: [%111], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                    %112 = arith.addi %32, %c3 : index
                    loom.copy %reinterpret_cast_4, %110 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%32, %c0], LR : [%112, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                    %113 = loom.bufferize_to_tensor %110[] : memref<f16> -> tensor<f16>
                    %114 = loom.sync ins(%113 : tensor<f16>) outs(%109 : tensor<f16>) -> tensor<f16>
                    loom.semaphore_give %110 : memref<f16>
                    %115 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %22, %54)
                    %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%115], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.copy %reinterpret_cast_5, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
                    %116 = loom.bufferize_to_tensor %82[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %117 = loom.sync ins(%116 : tensor<64x32xf16>) outs(%81 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %82 : memref<64x32xf16>
                    %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %117, %114 : tensor<64x32xf16>, tensor<64x32xf16>, tensor<f16>) outs(%81 : tensor<64x32xf16>) {
                    ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
                      %122 = arith.mulf %in_7, %in_8 : f16
                      %123 = arith.addf %in, %122 : f16
                      linalg.yield %123 : f16
                    } -> tensor<64x32xf16>
                    loom.semaphore_give %108 : memref<f16>
                    loom.semaphore_give %69 : memref<64x32xf16>
                    %119 = loom.sync ins(%118 : tensor<64x32xf16>) outs(%79 : tensor<64x32xf16>) -> tensor<64x32xf16>
                    loom.semaphore_give %80 : memref<64x32xf16>
                    %120 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 8388608 + d1 * 4096 + d2 * 64 + d3)>(%arg11, %45, %22, %54)
                    %reinterpret_cast_6 = memref.reinterpret_cast %arg7 to offset: [%120], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    %121 = loom.bufferize_to_memref %119 : tensor<64x32xf16> -> memref<64x32xf16>
                    loom.copy %121, %reinterpret_cast_6 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %63], LR : [%33, %63]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
                    loom.semaphore_give %78 : memref<64x32xf16>
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
