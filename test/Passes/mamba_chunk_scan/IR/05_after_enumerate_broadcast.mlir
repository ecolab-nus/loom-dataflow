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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 64 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c64, %23 : index
      %27 = arith.ceildivui %c256, %20 : index
      %28 = arith.ceildivui %c64, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c4 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg12, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %57 = loom.semaphore_take %56 : memref<?xf16> -> memref<?xf16>
                          %58 = loom.init_tensor %57[%20] : memref<?xf16> -> tensor<?xf16>
                          %59 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%55 : tensor<?xf16>) outs(%58 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %114 = arith.mulf %in, %cst_1 : f16
                            %115 = math.powf %cst, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?xf16>
                          %60 = arith.muli %43, %c256 : index
                          %61 = arith.divui %42, %c64 : index
                          %62 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %63 = loom.semaphore_take %62 : memref<?x64xf16> -> memref<?x64xf16>
                          %64 = loom.subview %arg4[%41, %60, %61, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %65 = arith.addi %48, %c1 : index
                          loom.copy %64, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %63[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = arith.muli %38, %21 : index
                          %68 = loom.alloc [%21, 64] on @L1 : memref<?x64xf16>
                          %69 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %70 = loom.subview %arg5[%41, %43, %42, %67, 0] [1, 1, 1, %21, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %71 = arith.addi %arg10, %50 : index
                          %72 = arith.addi %71, %51 : index
                          loom.copy %70, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %69[%21, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %75 = loom.semaphore_take %74 : memref<64x?xf16> -> memref<64x?xf16>
                          %76 = loom.init_tensor %75[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %transposed = linalg.transpose ins(%73 : tensor<?x64xf16>) outs(%76 : tensor<64x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %69 : memref<?x64xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = linalg.fill ins(%cst_0 : f16) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %83 = linalg.matmul ins(%66, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%82 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %75 : memref<64x?xf16>
                          loom.semaphore_give %63 : memref<?x64xf16>
                          %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%83, %59 : tensor<?x?xf16>, tensor<?xf16>) outs(%81 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %114 = arith.mulf %in, %in_2 : f16
                            linalg.yield %114 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          loom.semaphore_give %57 : memref<?xf16>
                          %85 = arith.addi %37, %c1 : index
                          %86 = arith.muli %85, %20 : index
                          %87 = arith.ceildivui %86, %22 : index
                          %88 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %89 = loom.semaphore_take %88 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %92 = loom.semaphore_take %91 : memref<?xf16> -> memref<?xf16>
                          %93 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %94 = loom.semaphore_take %93 : memref<?xf16> -> memref<?xf16>
                          %95 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = scf.for %arg18 = %c0 to %87 step %c1 iter_args(%arg19 = %84) -> (tensor<?x?xf16>) {
                            %114 = arith.muli %arg18, %22 : index
                            %115 = loom.subview %arg0[%41, %43, %61, %44, %114] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %115, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %116 = loom.bufferize_to_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %117 = loom.subview %arg1[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.subview %arg2[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %119, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %120 = loom.bufferize_to_tensor %94[%22] : memref<?xf16> -> tensor<?xf16>
                            %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %55, %118, %120 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%90 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %125 = arith.mulf %in_3, %cst_1 : f16
                              %126 = arith.mulf %in_2, %cst_1 : f16
                              %127 = arith.subf %126, %125 : f16
                              %128 = math.powf %cst, %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_4 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<?xf16>
                            loom.semaphore_give %92 : memref<?xf16>
                            %122 = loom.subview %arg3[%41, %60, %42, %67] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %96[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.matmul ins(%121, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            loom.semaphore_give %89 : memref<?x?xf16>
                            scf.yield %124 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %98 = loom.alloc [1] on @L1 : memref<f16>
                          %99 = loom.semaphore_take %98 : memref<f16> -> memref<f16>
                          %100 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %101 = arith.addi %51, %c3 : index
                          loom.copy %100, %99 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %101]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %102 = loom.bufferize_to_tensor %99[] : memref<f16> -> tensor<f16>
                          %103 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %104 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %105 = loom.init_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = loom.subview %arg3[%41, %60, %42, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %106, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %107 = loom.bufferize_to_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%97, %107, %102 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%105 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %114 = arith.mulf %in_2, %in_3 : f16
                            %115 = arith.addf %in, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %99 : memref<f16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          %109 = loom.subview %arg7[%41, %60, %42, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %110 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %111 = loom.init_tensor %110[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = loom.sync ins(%108 : tensor<?x?xf16>) outs(%111 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %113 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %113, %109 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %110 : memref<?x?xf16>
                          loom.semaphore_give %104 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 64 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c64, %23 : index
      %27 = arith.ceildivui %c256, %20 : index
      %28 = arith.ceildivui %c64, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c4 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg11, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %57 = loom.semaphore_take %56 : memref<?xf16> -> memref<?xf16>
                          %58 = loom.init_tensor %57[%20] : memref<?xf16> -> tensor<?xf16>
                          %59 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%55 : tensor<?xf16>) outs(%58 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %114 = arith.mulf %in, %cst_1 : f16
                            %115 = math.powf %cst, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?xf16>
                          %60 = arith.muli %43, %c256 : index
                          %61 = arith.divui %42, %c64 : index
                          %62 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %63 = loom.semaphore_take %62 : memref<?x64xf16> -> memref<?x64xf16>
                          %64 = loom.subview %arg4[%41, %60, %61, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %65 = arith.addi %48, %c1 : index
                          loom.copy %64, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %63[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = arith.muli %38, %21 : index
                          %68 = loom.alloc [%21, 64] on @L1 : memref<?x64xf16>
                          %69 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %70 = loom.subview %arg5[%41, %43, %42, %67, 0] [1, 1, 1, %21, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %71 = arith.addi %arg10, %50 : index
                          %72 = arith.addi %71, %51 : index
                          loom.copy %70, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %69[%21, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %75 = loom.semaphore_take %74 : memref<64x?xf16> -> memref<64x?xf16>
                          %76 = loom.init_tensor %75[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %transposed = linalg.transpose ins(%73 : tensor<?x64xf16>) outs(%76 : tensor<64x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %69 : memref<?x64xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = linalg.fill ins(%cst_0 : f16) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %83 = linalg.matmul ins(%66, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%82 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %75 : memref<64x?xf16>
                          loom.semaphore_give %63 : memref<?x64xf16>
                          %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%83, %59 : tensor<?x?xf16>, tensor<?xf16>) outs(%81 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %114 = arith.mulf %in, %in_2 : f16
                            linalg.yield %114 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          loom.semaphore_give %57 : memref<?xf16>
                          %85 = arith.addi %37, %c1 : index
                          %86 = arith.muli %85, %20 : index
                          %87 = arith.ceildivui %86, %22 : index
                          %88 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %89 = loom.semaphore_take %88 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %92 = loom.semaphore_take %91 : memref<?xf16> -> memref<?xf16>
                          %93 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %94 = loom.semaphore_take %93 : memref<?xf16> -> memref<?xf16>
                          %95 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = scf.for %arg18 = %c0 to %87 step %c1 iter_args(%arg19 = %84) -> (tensor<?x?xf16>) {
                            %114 = arith.muli %arg18, %22 : index
                            %115 = loom.subview %arg0[%41, %43, %61, %44, %114] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %115, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %116 = loom.bufferize_to_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %117 = loom.subview %arg1[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.subview %arg2[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %119, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %120 = loom.bufferize_to_tensor %94[%22] : memref<?xf16> -> tensor<?xf16>
                            %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %55, %118, %120 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%90 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %125 = arith.mulf %in_3, %cst_1 : f16
                              %126 = arith.mulf %in_2, %cst_1 : f16
                              %127 = arith.subf %126, %125 : f16
                              %128 = math.powf %cst, %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_4 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<?xf16>
                            loom.semaphore_give %92 : memref<?xf16>
                            %122 = loom.subview %arg3[%41, %60, %42, %67] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %96[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.matmul ins(%121, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            loom.semaphore_give %89 : memref<?x?xf16>
                            scf.yield %124 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %98 = loom.alloc [1] on @L1 : memref<f16>
                          %99 = loom.semaphore_take %98 : memref<f16> -> memref<f16>
                          %100 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %101 = arith.addi %51, %c3 : index
                          loom.copy %100, %99 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %101]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %102 = loom.bufferize_to_tensor %99[] : memref<f16> -> tensor<f16>
                          %103 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %104 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %105 = loom.init_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = loom.subview %arg3[%41, %60, %42, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %106, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %107 = loom.bufferize_to_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%97, %107, %102 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%105 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %114 = arith.mulf %in_2, %in_3 : f16
                            %115 = arith.addf %in, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %99 : memref<f16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          %109 = loom.subview %arg7[%41, %60, %42, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %110 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %111 = loom.init_tensor %110[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = loom.sync ins(%108 : tensor<?x?xf16>) outs(%111 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %113 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %113, %109 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %110 : memref<?x?xf16>
                          loom.semaphore_give %104 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 64 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c64, %23 : index
      %27 = arith.ceildivui %c256, %20 : index
      %28 = arith.ceildivui %c64, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c4 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c4 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg12, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %57 = loom.semaphore_take %56 : memref<?xf16> -> memref<?xf16>
                          %58 = loom.init_tensor %57[%20] : memref<?xf16> -> tensor<?xf16>
                          %59 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%55 : tensor<?xf16>) outs(%58 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %114 = arith.mulf %in, %cst_1 : f16
                            %115 = math.powf %cst, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?xf16>
                          %60 = arith.muli %43, %c256 : index
                          %61 = arith.divui %42, %c64 : index
                          %62 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %63 = loom.semaphore_take %62 : memref<?x64xf16> -> memref<?x64xf16>
                          %64 = loom.subview %arg4[%41, %60, %61, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %65 = arith.addi %48, %c3 : index
                          loom.copy %64, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %63[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = arith.muli %38, %21 : index
                          %68 = loom.alloc [%21, 64] on @L1 : memref<?x64xf16>
                          %69 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %70 = loom.subview %arg5[%41, %43, %42, %67, 0] [1, 1, 1, %21, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %71 = arith.addi %arg10, %50 : index
                          %72 = arith.addi %71, %51 : index
                          loom.copy %70, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %69[%21, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %75 = loom.semaphore_take %74 : memref<64x?xf16> -> memref<64x?xf16>
                          %76 = loom.init_tensor %75[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %transposed = linalg.transpose ins(%73 : tensor<?x64xf16>) outs(%76 : tensor<64x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %69 : memref<?x64xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = linalg.fill ins(%cst_0 : f16) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %83 = linalg.matmul ins(%66, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%82 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %75 : memref<64x?xf16>
                          loom.semaphore_give %63 : memref<?x64xf16>
                          %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%83, %59 : tensor<?x?xf16>, tensor<?xf16>) outs(%81 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %114 = arith.mulf %in, %in_2 : f16
                            linalg.yield %114 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          loom.semaphore_give %57 : memref<?xf16>
                          %85 = arith.addi %37, %c1 : index
                          %86 = arith.muli %85, %20 : index
                          %87 = arith.ceildivui %86, %22 : index
                          %88 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %89 = loom.semaphore_take %88 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %92 = loom.semaphore_take %91 : memref<?xf16> -> memref<?xf16>
                          %93 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %94 = loom.semaphore_take %93 : memref<?xf16> -> memref<?xf16>
                          %95 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = scf.for %arg18 = %c0 to %87 step %c1 iter_args(%arg19 = %84) -> (tensor<?x?xf16>) {
                            %114 = arith.muli %arg18, %22 : index
                            %115 = loom.subview %arg0[%41, %43, %61, %44, %114] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %115, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %116 = loom.bufferize_to_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %117 = loom.subview %arg1[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.subview %arg2[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %119, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %120 = loom.bufferize_to_tensor %94[%22] : memref<?xf16> -> tensor<?xf16>
                            %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %55, %118, %120 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%90 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %125 = arith.mulf %in_3, %cst_1 : f16
                              %126 = arith.mulf %in_2, %cst_1 : f16
                              %127 = arith.subf %126, %125 : f16
                              %128 = math.powf %cst, %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_4 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<?xf16>
                            loom.semaphore_give %92 : memref<?xf16>
                            %122 = loom.subview %arg3[%41, %60, %42, %67] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %96[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.matmul ins(%121, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            loom.semaphore_give %89 : memref<?x?xf16>
                            scf.yield %124 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %98 = loom.alloc [1] on @L1 : memref<f16>
                          %99 = loom.semaphore_take %98 : memref<f16> -> memref<f16>
                          %100 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %101 = arith.addi %51, %c3 : index
                          loom.copy %100, %99 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %101]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %102 = loom.bufferize_to_tensor %99[] : memref<f16> -> tensor<f16>
                          %103 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %104 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %105 = loom.init_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = loom.subview %arg3[%41, %60, %42, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %106, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %107 = loom.bufferize_to_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%97, %107, %102 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%105 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %114 = arith.mulf %in_2, %in_3 : f16
                            %115 = arith.addf %in, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %99 : memref<f16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          %109 = loom.subview %arg7[%41, %60, %42, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %110 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %111 = loom.init_tensor %110[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = loom.sync ins(%108 : tensor<?x?xf16>) outs(%111 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %113 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %113, %109 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %110 : memref<?x?xf16>
                          loom.semaphore_give %104 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 64 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c64, %23 : index
      %27 = arith.ceildivui %c256, %20 : index
      %28 = arith.ceildivui %c64, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c4 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c4 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg11, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %57 = loom.semaphore_take %56 : memref<?xf16> -> memref<?xf16>
                          %58 = loom.init_tensor %57[%20] : memref<?xf16> -> tensor<?xf16>
                          %59 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%55 : tensor<?xf16>) outs(%58 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %114 = arith.mulf %in, %cst_1 : f16
                            %115 = math.powf %cst, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?xf16>
                          %60 = arith.muli %43, %c256 : index
                          %61 = arith.divui %42, %c64 : index
                          %62 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %63 = loom.semaphore_take %62 : memref<?x64xf16> -> memref<?x64xf16>
                          %64 = loom.subview %arg4[%41, %60, %61, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %65 = arith.addi %48, %c3 : index
                          loom.copy %64, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %63[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = arith.muli %38, %21 : index
                          %68 = loom.alloc [%21, 64] on @L1 : memref<?x64xf16>
                          %69 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %70 = loom.subview %arg5[%41, %43, %42, %67, 0] [1, 1, 1, %21, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %71 = arith.addi %arg10, %50 : index
                          %72 = arith.addi %71, %51 : index
                          loom.copy %70, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %69[%21, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %75 = loom.semaphore_take %74 : memref<64x?xf16> -> memref<64x?xf16>
                          %76 = loom.init_tensor %75[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %transposed = linalg.transpose ins(%73 : tensor<?x64xf16>) outs(%76 : tensor<64x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %69 : memref<?x64xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = linalg.fill ins(%cst_0 : f16) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %83 = linalg.matmul ins(%66, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%82 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %75 : memref<64x?xf16>
                          loom.semaphore_give %63 : memref<?x64xf16>
                          %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%83, %59 : tensor<?x?xf16>, tensor<?xf16>) outs(%81 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %114 = arith.mulf %in, %in_2 : f16
                            linalg.yield %114 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          loom.semaphore_give %57 : memref<?xf16>
                          %85 = arith.addi %37, %c1 : index
                          %86 = arith.muli %85, %20 : index
                          %87 = arith.ceildivui %86, %22 : index
                          %88 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %89 = loom.semaphore_take %88 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %92 = loom.semaphore_take %91 : memref<?xf16> -> memref<?xf16>
                          %93 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %94 = loom.semaphore_take %93 : memref<?xf16> -> memref<?xf16>
                          %95 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = scf.for %arg18 = %c0 to %87 step %c1 iter_args(%arg19 = %84) -> (tensor<?x?xf16>) {
                            %114 = arith.muli %arg18, %22 : index
                            %115 = loom.subview %arg0[%41, %43, %61, %44, %114] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %115, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %116 = loom.bufferize_to_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %117 = loom.subview %arg1[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.subview %arg2[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %119, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%65, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %120 = loom.bufferize_to_tensor %94[%22] : memref<?xf16> -> tensor<?xf16>
                            %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %55, %118, %120 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%90 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %125 = arith.mulf %in_3, %cst_1 : f16
                              %126 = arith.mulf %in_2, %cst_1 : f16
                              %127 = arith.subf %126, %125 : f16
                              %128 = math.powf %cst, %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_4 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<?xf16>
                            loom.semaphore_give %92 : memref<?xf16>
                            %122 = loom.subview %arg3[%41, %60, %42, %67] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %96[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.matmul ins(%121, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            loom.semaphore_give %89 : memref<?x?xf16>
                            scf.yield %124 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %98 = loom.alloc [1] on @L1 : memref<f16>
                          %99 = loom.semaphore_take %98 : memref<f16> -> memref<f16>
                          %100 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %101 = arith.addi %51, %c3 : index
                          loom.copy %100, %99 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %101]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %102 = loom.bufferize_to_tensor %99[] : memref<f16> -> tensor<f16>
                          %103 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %104 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %105 = loom.init_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = loom.subview %arg3[%41, %60, %42, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %106, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %107 = loom.bufferize_to_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%97, %107, %102 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%105 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %114 = arith.mulf %in_2, %in_3 : f16
                            %115 = arith.addf %in, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %99 : memref<f16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          %109 = loom.subview %arg7[%41, %60, %42, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %110 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %111 = loom.init_tensor %110[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = loom.sync ins(%108 : tensor<?x?xf16>) outs(%111 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %113 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %113, %109 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%48, %72], LR : [%65, %72]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %110 : memref<?x?xf16>
                          loom.semaphore_give %104 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 64 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c64, %23 : index
      %27 = arith.ceildivui %c256, %20 : index
      %28 = arith.ceildivui %c64, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c4 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg12, %c2 : index
                          %53 = arith.addi %52, %c1 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %56 = loom.semaphore_take %55 : memref<?xf16> -> memref<?xf16>
                          %57 = loom.init_tensor %56[%20] : memref<?xf16> -> tensor<?xf16>
                          %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%57 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %114 = arith.mulf %in, %cst_1 : f16
                            %115 = math.powf %cst, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?xf16>
                          %59 = arith.muli %43, %c256 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %64 = arith.addi %48, %50 : index
                          %65 = arith.addi %48, %c1 : index
                          %66 = arith.addi %65, %50 : index
                          loom.copy %63, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %67 = loom.bufferize_to_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [%21, 64] on @L1 : memref<?x64xf16>
                          %70 = loom.semaphore_take %69 : memref<?x64xf16> -> memref<?x64xf16>
                          %71 = loom.subview %arg5[%41, %43, %42, %68, 0] [1, 1, 1, %21, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %72 = arith.addi %arg10, %52 : index
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %70[%21, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %75 = loom.semaphore_take %74 : memref<64x?xf16> -> memref<64x?xf16>
                          %76 = loom.init_tensor %75[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %transposed = linalg.transpose ins(%73 : tensor<?x64xf16>) outs(%76 : tensor<64x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %70 : memref<?x64xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = linalg.fill ins(%cst_0 : f16) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %83 = linalg.matmul ins(%67, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%82 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %75 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%83, %58 : tensor<?x?xf16>, tensor<?xf16>) outs(%81 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %114 = arith.mulf %in, %in_2 : f16
                            linalg.yield %114 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          loom.semaphore_give %56 : memref<?xf16>
                          %85 = arith.addi %37, %c1 : index
                          %86 = arith.muli %85, %20 : index
                          %87 = arith.ceildivui %86, %22 : index
                          %88 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %89 = loom.semaphore_take %88 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %92 = loom.semaphore_take %91 : memref<?xf16> -> memref<?xf16>
                          %93 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %94 = loom.semaphore_take %93 : memref<?xf16> -> memref<?xf16>
                          %95 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = scf.for %arg18 = %c0 to %87 step %c1 iter_args(%arg19 = %84) -> (tensor<?x?xf16>) {
                            %114 = arith.muli %arg18, %22 : index
                            %115 = loom.subview %arg0[%41, %43, %60, %44, %114] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %115, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %116 = loom.bufferize_to_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %117 = loom.subview %arg1[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.subview %arg2[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %119, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %120 = loom.bufferize_to_tensor %94[%22] : memref<?xf16> -> tensor<?xf16>
                            %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %54, %118, %120 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%90 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %125 = arith.mulf %in_3, %cst_1 : f16
                              %126 = arith.mulf %in_2, %cst_1 : f16
                              %127 = arith.subf %126, %125 : f16
                              %128 = math.powf %cst, %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_4 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<?xf16>
                            loom.semaphore_give %92 : memref<?xf16>
                            %122 = loom.subview %arg3[%41, %59, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %96[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.matmul ins(%121, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            loom.semaphore_give %89 : memref<?x?xf16>
                            scf.yield %124 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %98 = loom.alloc [1] on @L1 : memref<f16>
                          %99 = loom.semaphore_take %98 : memref<f16> -> memref<f16>
                          %100 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %101 = arith.addi %50, %c3 : index
                          loom.copy %100, %99 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%101, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %102 = loom.bufferize_to_tensor %99[] : memref<f16> -> tensor<f16>
                          %103 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %104 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %105 = loom.init_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %106, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %107 = loom.bufferize_to_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%97, %107, %102 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%105 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %114 = arith.mulf %in_2, %in_3 : f16
                            %115 = arith.addf %in, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %99 : memref<f16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          %109 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %110 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %111 = loom.init_tensor %110[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = loom.sync ins(%108 : tensor<?x?xf16>) outs(%111 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %113 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %113, %109 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %110 : memref<?x?xf16>
                          loom.semaphore_give %104 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 64 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c64, %23 : index
      %27 = arith.ceildivui %c256, %20 : index
      %28 = arith.ceildivui %c64, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c2 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c4 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg11, %c2 : index
                          %53 = arith.addi %52, %c1 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %56 = loom.semaphore_take %55 : memref<?xf16> -> memref<?xf16>
                          %57 = loom.init_tensor %56[%20] : memref<?xf16> -> tensor<?xf16>
                          %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%57 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %114 = arith.mulf %in, %cst_1 : f16
                            %115 = math.powf %cst, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?xf16>
                          %59 = arith.muli %43, %c256 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %64 = arith.addi %48, %50 : index
                          %65 = arith.addi %48, %c1 : index
                          %66 = arith.addi %65, %50 : index
                          loom.copy %63, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %67 = loom.bufferize_to_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [%21, 64] on @L1 : memref<?x64xf16>
                          %70 = loom.semaphore_take %69 : memref<?x64xf16> -> memref<?x64xf16>
                          %71 = loom.subview %arg5[%41, %43, %42, %68, 0] [1, 1, 1, %21, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %72 = arith.addi %arg10, %52 : index
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %70[%21, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %75 = loom.semaphore_take %74 : memref<64x?xf16> -> memref<64x?xf16>
                          %76 = loom.init_tensor %75[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %transposed = linalg.transpose ins(%73 : tensor<?x64xf16>) outs(%76 : tensor<64x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %70 : memref<?x64xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = linalg.fill ins(%cst_0 : f16) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %83 = linalg.matmul ins(%67, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%82 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %75 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%83, %58 : tensor<?x?xf16>, tensor<?xf16>) outs(%81 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %114 = arith.mulf %in, %in_2 : f16
                            linalg.yield %114 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          loom.semaphore_give %56 : memref<?xf16>
                          %85 = arith.addi %37, %c1 : index
                          %86 = arith.muli %85, %20 : index
                          %87 = arith.ceildivui %86, %22 : index
                          %88 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %89 = loom.semaphore_take %88 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %92 = loom.semaphore_take %91 : memref<?xf16> -> memref<?xf16>
                          %93 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %94 = loom.semaphore_take %93 : memref<?xf16> -> memref<?xf16>
                          %95 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = scf.for %arg18 = %c0 to %87 step %c1 iter_args(%arg19 = %84) -> (tensor<?x?xf16>) {
                            %114 = arith.muli %arg18, %22 : index
                            %115 = loom.subview %arg0[%41, %43, %60, %44, %114] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %115, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %116 = loom.bufferize_to_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %117 = loom.subview %arg1[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.subview %arg2[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %119, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %120 = loom.bufferize_to_tensor %94[%22] : memref<?xf16> -> tensor<?xf16>
                            %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %54, %118, %120 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%90 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %125 = arith.mulf %in_3, %cst_1 : f16
                              %126 = arith.mulf %in_2, %cst_1 : f16
                              %127 = arith.subf %126, %125 : f16
                              %128 = math.powf %cst, %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_4 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<?xf16>
                            loom.semaphore_give %92 : memref<?xf16>
                            %122 = loom.subview %arg3[%41, %59, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %96[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.matmul ins(%121, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            loom.semaphore_give %89 : memref<?x?xf16>
                            scf.yield %124 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %98 = loom.alloc [1] on @L1 : memref<f16>
                          %99 = loom.semaphore_take %98 : memref<f16> -> memref<f16>
                          %100 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %101 = arith.addi %50, %c3 : index
                          loom.copy %100, %99 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%101, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %102 = loom.bufferize_to_tensor %99[] : memref<f16> -> tensor<f16>
                          %103 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %104 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %105 = loom.init_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %106, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %107 = loom.bufferize_to_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%97, %107, %102 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%105 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %114 = arith.mulf %in_2, %in_3 : f16
                            %115 = arith.addf %in, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %99 : memref<f16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          %109 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %110 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %111 = loom.init_tensor %110[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = loom.sync ins(%108 : tensor<?x?xf16>) outs(%111 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %113 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %113, %109 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %110 : memref<?x?xf16>
                          loom.semaphore_give %104 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 64 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c64, %23 : index
      %27 = arith.ceildivui %c256, %20 : index
      %28 = arith.ceildivui %c64, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c4 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg12, %c4 : index
                          %53 = arith.addi %52, %c3 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %56 = loom.semaphore_take %55 : memref<?xf16> -> memref<?xf16>
                          %57 = loom.init_tensor %56[%20] : memref<?xf16> -> tensor<?xf16>
                          %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%57 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %114 = arith.mulf %in, %cst_1 : f16
                            %115 = math.powf %cst, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?xf16>
                          %59 = arith.muli %43, %c256 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %64 = arith.addi %48, %50 : index
                          %65 = arith.addi %48, %c1 : index
                          %66 = arith.addi %65, %50 : index
                          loom.copy %63, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %67 = loom.bufferize_to_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [%21, 64] on @L1 : memref<?x64xf16>
                          %70 = loom.semaphore_take %69 : memref<?x64xf16> -> memref<?x64xf16>
                          %71 = loom.subview %arg5[%41, %43, %42, %68, 0] [1, 1, 1, %21, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %72 = arith.addi %arg10, %52 : index
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %70[%21, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %75 = loom.semaphore_take %74 : memref<64x?xf16> -> memref<64x?xf16>
                          %76 = loom.init_tensor %75[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %transposed = linalg.transpose ins(%73 : tensor<?x64xf16>) outs(%76 : tensor<64x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %70 : memref<?x64xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = linalg.fill ins(%cst_0 : f16) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %83 = linalg.matmul ins(%67, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%82 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %75 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%83, %58 : tensor<?x?xf16>, tensor<?xf16>) outs(%81 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %114 = arith.mulf %in, %in_2 : f16
                            linalg.yield %114 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          loom.semaphore_give %56 : memref<?xf16>
                          %85 = arith.addi %37, %c1 : index
                          %86 = arith.muli %85, %20 : index
                          %87 = arith.ceildivui %86, %22 : index
                          %88 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %89 = loom.semaphore_take %88 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %92 = loom.semaphore_take %91 : memref<?xf16> -> memref<?xf16>
                          %93 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %94 = loom.semaphore_take %93 : memref<?xf16> -> memref<?xf16>
                          %95 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = scf.for %arg18 = %c0 to %87 step %c1 iter_args(%arg19 = %84) -> (tensor<?x?xf16>) {
                            %114 = arith.muli %arg18, %22 : index
                            %115 = loom.subview %arg0[%41, %43, %60, %44, %114] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %115, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %116 = loom.bufferize_to_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %117 = loom.subview %arg1[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.subview %arg2[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %119, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %120 = loom.bufferize_to_tensor %94[%22] : memref<?xf16> -> tensor<?xf16>
                            %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %54, %118, %120 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%90 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %125 = arith.mulf %in_3, %cst_1 : f16
                              %126 = arith.mulf %in_2, %cst_1 : f16
                              %127 = arith.subf %126, %125 : f16
                              %128 = math.powf %cst, %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_4 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<?xf16>
                            loom.semaphore_give %92 : memref<?xf16>
                            %122 = loom.subview %arg3[%41, %59, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %96[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.matmul ins(%121, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            loom.semaphore_give %89 : memref<?x?xf16>
                            scf.yield %124 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %98 = loom.alloc [1] on @L1 : memref<f16>
                          %99 = loom.semaphore_take %98 : memref<f16> -> memref<f16>
                          %100 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %101 = arith.addi %50, %c3 : index
                          loom.copy %100, %99 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%101, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %102 = loom.bufferize_to_tensor %99[] : memref<f16> -> tensor<f16>
                          %103 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %104 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %105 = loom.init_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %106, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %107 = loom.bufferize_to_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%97, %107, %102 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%105 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %114 = arith.mulf %in_2, %in_3 : f16
                            %115 = arith.addf %in, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %99 : memref<f16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          %109 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %110 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %111 = loom.init_tensor %110[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = loom.sync ins(%108 : tensor<?x?xf16>) outs(%111 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %113 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %113, %109 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %110 : memref<?x?xf16>
                          loom.semaphore_give %104 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c256 = arith.constant 256 : index
      %c64 = arith.constant 64 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 64 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c64, %23 : index
      %27 = arith.ceildivui %c256, %20 : index
      %28 = arith.ceildivui %c64, %21 : index
      %29 = arith.ceildivui %c2, %24 : index
      %30 = arith.ceildivui %c8, %25 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %31 = arith.ceildivui %26, %c2 : index
                scf.for %arg13 = %c0 to %31 step %c1 {
                  %32 = arith.ceildivui %27, %c2 : index
                  scf.for %arg14 = %c0 to %32 step %c1 {
                    %33 = arith.ceildivui %28, %c4 : index
                    scf.for %arg15 = %c0 to %33 step %c1 {
                      %34 = arith.ceildivui %29, %c2 : index
                      scf.for %arg16 = %c0 to %34 step %c1 {
                        %35 = arith.ceildivui %30, %c2 : index
                        scf.for %arg17 = %c0 to %35 step %c1 {
                          %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                          %37 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                          %38 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg10, %arg15)
                          %39 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg11, %arg16)
                          %40 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg17)
                          %41 = arith.muli %39, %24 : index
                          %42 = arith.muli %36, %23 : index
                          %43 = arith.muli %40, %25 : index
                          %44 = arith.muli %37, %20 : index
                          %45 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %46 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg11, %c4 : index
                          %53 = arith.addi %52, %c3 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %56 = loom.semaphore_take %55 : memref<?xf16> -> memref<?xf16>
                          %57 = loom.init_tensor %56[%20] : memref<?xf16> -> tensor<?xf16>
                          %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%57 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %114 = arith.mulf %in, %cst_1 : f16
                            %115 = math.powf %cst, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?xf16>
                          %59 = arith.muli %43, %c256 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %64 = arith.addi %48, %50 : index
                          %65 = arith.addi %48, %c1 : index
                          %66 = arith.addi %65, %50 : index
                          loom.copy %63, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %67 = loom.bufferize_to_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [%21, 64] on @L1 : memref<?x64xf16>
                          %70 = loom.semaphore_take %69 : memref<?x64xf16> -> memref<?x64xf16>
                          %71 = loom.subview %arg5[%41, %43, %42, %68, 0] [1, 1, 1, %21, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          %72 = arith.addi %arg10, %52 : index
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %70[%21, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %75 = loom.semaphore_take %74 : memref<64x?xf16> -> memref<64x?xf16>
                          %76 = loom.init_tensor %75[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %transposed = linalg.transpose ins(%73 : tensor<?x64xf16>) outs(%76 : tensor<64x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %70 : memref<?x64xf16>
                          %77 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %79 = loom.init_tensor %78[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %80 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                          %81 = loom.init_tensor %80[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %82 = linalg.fill ins(%cst_0 : f16) outs(%79 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %83 = linalg.matmul ins(%67, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%82 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %75 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%83, %58 : tensor<?x?xf16>, tensor<?xf16>) outs(%81 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %114 = arith.mulf %in, %in_2 : f16
                            linalg.yield %114 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          loom.semaphore_give %56 : memref<?xf16>
                          %85 = arith.addi %37, %c1 : index
                          %86 = arith.muli %85, %20 : index
                          %87 = arith.ceildivui %86, %22 : index
                          %88 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %89 = loom.semaphore_take %88 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %92 = loom.semaphore_take %91 : memref<?xf16> -> memref<?xf16>
                          %93 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %94 = loom.semaphore_take %93 : memref<?xf16> -> memref<?xf16>
                          %95 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = scf.for %arg18 = %c0 to %87 step %c1 iter_args(%arg19 = %84) -> (tensor<?x?xf16>) {
                            %114 = arith.muli %arg18, %22 : index
                            %115 = loom.subview %arg0[%41, %43, %60, %44, %114] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %115, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %116 = loom.bufferize_to_tensor %89[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %117 = loom.subview %arg1[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.subview %arg2[%41, %42, %43, %114] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %119, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%64, %52], LR : [%66, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %120 = loom.bufferize_to_tensor %94[%22] : memref<?xf16> -> tensor<?xf16>
                            %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %54, %118, %120 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%90 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %125 = arith.mulf %in_3, %cst_1 : f16
                              %126 = arith.mulf %in_2, %cst_1 : f16
                              %127 = arith.subf %126, %125 : f16
                              %128 = math.powf %cst, %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_4 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %94 : memref<?xf16>
                            loom.semaphore_give %92 : memref<?xf16>
                            %122 = loom.subview %arg3[%41, %59, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %96[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.matmul ins(%121, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            loom.semaphore_give %89 : memref<?x?xf16>
                            scf.yield %124 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %98 = loom.alloc [1] on @L1 : memref<f16>
                          %99 = loom.semaphore_take %98 : memref<f16> -> memref<f16>
                          %100 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %101 = arith.addi %50, %c3 : index
                          loom.copy %100, %99 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%101, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %102 = loom.bufferize_to_tensor %99[] : memref<f16> -> tensor<f16>
                          %103 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %104 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %105 = loom.init_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %106, %104 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %107 = loom.bufferize_to_tensor %104[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%97, %107, %102 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%105 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %114 = arith.mulf %in_2, %in_3 : f16
                            %115 = arith.addf %in, %114 : f16
                            linalg.yield %115 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %99 : memref<f16>
                          loom.semaphore_give %80 : memref<?x?xf16>
                          %109 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %110 = loom.semaphore_take %103 : memref<?x?xf16> -> memref<?x?xf16>
                          %111 = loom.init_tensor %110[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %112 = loom.sync ins(%108 : tensor<?x?xf16>) outs(%111 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %113 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %113, %109 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%64, %72], LR : [%66, %72]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %110 : memref<?x?xf16>
                          loom.semaphore_give %104 : memref<?x?xf16>
                        } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
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
