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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %50 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %51 = loom.init_tensor %50[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %52 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %53 = loom.init_tensor %52[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %54 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%53 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %55 = arith.muli %43, %c256 : index
                          %56 = arith.addi %44, %55 : index
                          %57 = arith.divui %42, %c64 : index
                          %58 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %59 = loom.semaphore_take %58 : memref<?x64xf16> -> memref<?x64xf16>
                          %60 = loom.subview %arg4[%41, %56, %57, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %60, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %61 = loom.bufferize_to_tensor %59[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %62 = arith.muli %38, %21 : index
                          %63 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %64 = loom.semaphore_take %63 : memref<64x?xf16> -> memref<64x?xf16>
                          %65 = loom.subview %arg5[%41, %43, %42, 0, %62] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %66 = loom.bufferize_to_tensor %64[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %73 = loom.init_tensor %72[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %74 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %75 = linalg.matmul ins(%61, %66 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%74 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %64 : memref<64x?xf16>
                          loom.semaphore_give %59 : memref<?x64xf16>
                          %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %110 = math.exp %in_0 : f16
                            %111 = arith.mulf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          loom.semaphore_give %52 : memref<?x32xf16>
                          %77 = arith.addi %37, %c1 : index
                          %78 = arith.muli %77, %20 : index
                          %79 = arith.ceildivui %78, %22 : index
                          %80 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %81 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %82 = loom.init_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %83 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %84 = loom.init_tensor %83[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %85 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = loom.init_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %88 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %89 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %90 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %91 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %92 = loom.semaphore_take %91 : memref<32x?xf16> -> memref<32x?xf16>
                          %93 = loom.init_tensor %92[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %94 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %95 = loom.semaphore_take %94 : memref<32x?xf16> -> memref<32x?xf16>
                          %96 = loom.init_tensor %95[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %97 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = scf.for %arg18 = %c0 to %79 step %c1 iter_args(%arg19 = %76) -> (tensor<?x?xf16>) {
                            %110 = arith.muli %arg18, %22 : index
                            %111 = loom.subview %arg0[%41, %43, %57, %44, %110] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %111, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %112 = loom.bufferize_to_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %113 = loom.subview %arg1[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %113, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %114 = loom.bufferize_to_tensor %90[%22] : memref<?xf16> -> tensor<?xf16>
                            %115 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%51 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %116 = loom.broadcast ins(%114 : tensor<?xf16>) outs(%93 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %90 : memref<?xf16>
                            %117 = loom.subview %arg2[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %89[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.broadcast ins(%118 : tensor<?xf16>) outs(%96 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %89 : memref<?xf16>
                            %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %115, %116, %119 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%87 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %127 = arith.subf %in_0, %in_1 : f16
                              %128 = math.exp %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_2 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %95 : memref<32x?xf16>
                            loom.semaphore_give %92 : memref<32x?xf16>
                            loom.semaphore_give %50 : memref<?x32xf16>
                            %121 = arith.addi %110, %55 : index
                            %122 = loom.subview %arg3[%41, %121, %42, %62] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %98[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.fill ins(%cst : f16) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %125 = linalg.matmul ins(%120, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %125 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %127 = arith.addf %in, %in_0 : f16
                              linalg.yield %127 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %83 : memref<?x?xf16>
                            scf.yield %126 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %100 = loom.alloc [1] on @L1 : memref<f16>
                          %101 = loom.semaphore_take %100 : memref<f16> -> memref<f16>
                          %102 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %102, %101 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %103 = loom.bufferize_to_tensor %101[] : memref<f16> -> tensor<f16>
                          %104 = loom.subview %arg3[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %104, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %105 = loom.bufferize_to_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %105, %103 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%82 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %110 = arith.mulf %in_0, %in_1 : f16
                            %111 = arith.addf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %101 : memref<f16>
                          loom.semaphore_give %72 : memref<?x?xf16>
                          %107 = loom.sync ins(%106 : tensor<?x?xf16>) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %81 : memref<?x?xf16>
                          %108 = loom.subview %arg7[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %109 = loom.bufferize_to_memref %107 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %109, %108 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %68 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %50 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %51 = loom.init_tensor %50[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %52 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %53 = loom.init_tensor %52[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %54 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%53 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %55 = arith.muli %43, %c256 : index
                          %56 = arith.addi %44, %55 : index
                          %57 = arith.divui %42, %c64 : index
                          %58 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %59 = loom.semaphore_take %58 : memref<?x64xf16> -> memref<?x64xf16>
                          %60 = loom.subview %arg4[%41, %56, %57, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %60, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %61 = loom.bufferize_to_tensor %59[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %62 = arith.muli %38, %21 : index
                          %63 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %64 = loom.semaphore_take %63 : memref<64x?xf16> -> memref<64x?xf16>
                          %65 = loom.subview %arg5[%41, %43, %42, 0, %62] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %66 = loom.bufferize_to_tensor %64[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %73 = loom.init_tensor %72[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %74 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %75 = linalg.matmul ins(%61, %66 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%74 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %64 : memref<64x?xf16>
                          loom.semaphore_give %59 : memref<?x64xf16>
                          %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %110 = math.exp %in_0 : f16
                            %111 = arith.mulf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          loom.semaphore_give %52 : memref<?x32xf16>
                          %77 = arith.addi %37, %c1 : index
                          %78 = arith.muli %77, %20 : index
                          %79 = arith.ceildivui %78, %22 : index
                          %80 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %81 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %82 = loom.init_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %83 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %84 = loom.init_tensor %83[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %85 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = loom.init_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %88 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %89 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %90 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %91 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %92 = loom.semaphore_take %91 : memref<32x?xf16> -> memref<32x?xf16>
                          %93 = loom.init_tensor %92[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %94 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %95 = loom.semaphore_take %94 : memref<32x?xf16> -> memref<32x?xf16>
                          %96 = loom.init_tensor %95[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %97 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = scf.for %arg18 = %c0 to %79 step %c1 iter_args(%arg19 = %76) -> (tensor<?x?xf16>) {
                            %110 = arith.muli %arg18, %22 : index
                            %111 = loom.subview %arg0[%41, %43, %57, %44, %110] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %111, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %112 = loom.bufferize_to_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %113 = loom.subview %arg1[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %113, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %114 = loom.bufferize_to_tensor %90[%22] : memref<?xf16> -> tensor<?xf16>
                            %115 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%51 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %116 = loom.broadcast ins(%114 : tensor<?xf16>) outs(%93 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %90 : memref<?xf16>
                            %117 = loom.subview %arg2[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %89[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.broadcast ins(%118 : tensor<?xf16>) outs(%96 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %89 : memref<?xf16>
                            %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %115, %116, %119 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%87 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %127 = arith.subf %in_0, %in_1 : f16
                              %128 = math.exp %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_2 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %95 : memref<32x?xf16>
                            loom.semaphore_give %92 : memref<32x?xf16>
                            loom.semaphore_give %50 : memref<?x32xf16>
                            %121 = arith.addi %110, %55 : index
                            %122 = loom.subview %arg3[%41, %121, %42, %62] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %98[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.fill ins(%cst : f16) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %125 = linalg.matmul ins(%120, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %125 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %127 = arith.addf %in, %in_0 : f16
                              linalg.yield %127 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %83 : memref<?x?xf16>
                            scf.yield %126 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %100 = loom.alloc [1] on @L1 : memref<f16>
                          %101 = loom.semaphore_take %100 : memref<f16> -> memref<f16>
                          %102 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %102, %101 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %103 = loom.bufferize_to_tensor %101[] : memref<f16> -> tensor<f16>
                          %104 = loom.subview %arg3[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %104, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %105 = loom.bufferize_to_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %105, %103 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%82 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %110 = arith.mulf %in_0, %in_1 : f16
                            %111 = arith.addf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %101 : memref<f16>
                          loom.semaphore_give %72 : memref<?x?xf16>
                          %107 = loom.sync ins(%106 : tensor<?x?xf16>) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %81 : memref<?x?xf16>
                          %108 = loom.subview %arg7[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %109 = loom.bufferize_to_memref %107 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %109, %108 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %68 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %50 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %51 = loom.init_tensor %50[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %52 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %53 = loom.init_tensor %52[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %54 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%53 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %55 = arith.muli %43, %c256 : index
                          %56 = arith.addi %44, %55 : index
                          %57 = arith.divui %42, %c64 : index
                          %58 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %59 = loom.semaphore_take %58 : memref<?x64xf16> -> memref<?x64xf16>
                          %60 = loom.subview %arg4[%41, %56, %57, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %60, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %61 = loom.bufferize_to_tensor %59[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %62 = arith.muli %38, %21 : index
                          %63 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %64 = loom.semaphore_take %63 : memref<64x?xf16> -> memref<64x?xf16>
                          %65 = loom.subview %arg5[%41, %43, %42, 0, %62] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %66 = loom.bufferize_to_tensor %64[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %73 = loom.init_tensor %72[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %74 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %75 = linalg.matmul ins(%61, %66 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%74 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %64 : memref<64x?xf16>
                          loom.semaphore_give %59 : memref<?x64xf16>
                          %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %110 = math.exp %in_0 : f16
                            %111 = arith.mulf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          loom.semaphore_give %52 : memref<?x32xf16>
                          %77 = arith.addi %37, %c1 : index
                          %78 = arith.muli %77, %20 : index
                          %79 = arith.ceildivui %78, %22 : index
                          %80 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %81 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %82 = loom.init_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %83 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %84 = loom.init_tensor %83[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %85 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = loom.init_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %88 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %89 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %90 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %91 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %92 = loom.semaphore_take %91 : memref<32x?xf16> -> memref<32x?xf16>
                          %93 = loom.init_tensor %92[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %94 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %95 = loom.semaphore_take %94 : memref<32x?xf16> -> memref<32x?xf16>
                          %96 = loom.init_tensor %95[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %97 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = scf.for %arg18 = %c0 to %79 step %c1 iter_args(%arg19 = %76) -> (tensor<?x?xf16>) {
                            %110 = arith.muli %arg18, %22 : index
                            %111 = loom.subview %arg0[%41, %43, %57, %44, %110] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %111, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %112 = loom.bufferize_to_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %113 = loom.subview %arg1[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %113, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %114 = loom.bufferize_to_tensor %90[%22] : memref<?xf16> -> tensor<?xf16>
                            %115 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%51 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %116 = loom.broadcast ins(%114 : tensor<?xf16>) outs(%93 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %90 : memref<?xf16>
                            %117 = loom.subview %arg2[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %89[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.broadcast ins(%118 : tensor<?xf16>) outs(%96 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %89 : memref<?xf16>
                            %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %115, %116, %119 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%87 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %127 = arith.subf %in_0, %in_1 : f16
                              %128 = math.exp %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_2 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %95 : memref<32x?xf16>
                            loom.semaphore_give %92 : memref<32x?xf16>
                            loom.semaphore_give %50 : memref<?x32xf16>
                            %121 = arith.addi %110, %55 : index
                            %122 = loom.subview %arg3[%41, %121, %42, %62] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %98[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.fill ins(%cst : f16) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %125 = linalg.matmul ins(%120, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %125 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %127 = arith.addf %in, %in_0 : f16
                              linalg.yield %127 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %83 : memref<?x?xf16>
                            scf.yield %126 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %100 = loom.alloc [1] on @L1 : memref<f16>
                          %101 = loom.semaphore_take %100 : memref<f16> -> memref<f16>
                          %102 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %102, %101 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %103 = loom.bufferize_to_tensor %101[] : memref<f16> -> tensor<f16>
                          %104 = loom.subview %arg3[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %104, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %105 = loom.bufferize_to_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %105, %103 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%82 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %110 = arith.mulf %in_0, %in_1 : f16
                            %111 = arith.addf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %101 : memref<f16>
                          loom.semaphore_give %72 : memref<?x?xf16>
                          %107 = loom.sync ins(%106 : tensor<?x?xf16>) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %81 : memref<?x?xf16>
                          %108 = loom.subview %arg7[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %109 = loom.bufferize_to_memref %107 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %109, %108 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %68 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %50 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %51 = loom.init_tensor %50[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %52 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %53 = loom.init_tensor %52[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %54 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%53 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %55 = arith.muli %43, %c256 : index
                          %56 = arith.addi %44, %55 : index
                          %57 = arith.divui %42, %c64 : index
                          %58 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %59 = loom.semaphore_take %58 : memref<?x64xf16> -> memref<?x64xf16>
                          %60 = loom.subview %arg4[%41, %56, %57, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %60, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %61 = loom.bufferize_to_tensor %59[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %62 = arith.muli %38, %21 : index
                          %63 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %64 = loom.semaphore_take %63 : memref<64x?xf16> -> memref<64x?xf16>
                          %65 = loom.subview %arg5[%41, %43, %42, 0, %62] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %66 = loom.bufferize_to_tensor %64[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %73 = loom.init_tensor %72[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %74 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %75 = linalg.matmul ins(%61, %66 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%74 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %64 : memref<64x?xf16>
                          loom.semaphore_give %59 : memref<?x64xf16>
                          %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %110 = math.exp %in_0 : f16
                            %111 = arith.mulf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          loom.semaphore_give %52 : memref<?x32xf16>
                          %77 = arith.addi %37, %c1 : index
                          %78 = arith.muli %77, %20 : index
                          %79 = arith.ceildivui %78, %22 : index
                          %80 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %81 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %82 = loom.init_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %83 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %84 = loom.init_tensor %83[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %85 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = loom.init_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %88 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %89 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %90 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %91 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %92 = loom.semaphore_take %91 : memref<32x?xf16> -> memref<32x?xf16>
                          %93 = loom.init_tensor %92[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %94 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %95 = loom.semaphore_take %94 : memref<32x?xf16> -> memref<32x?xf16>
                          %96 = loom.init_tensor %95[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %97 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = scf.for %arg18 = %c0 to %79 step %c1 iter_args(%arg19 = %76) -> (tensor<?x?xf16>) {
                            %110 = arith.muli %arg18, %22 : index
                            %111 = loom.subview %arg0[%41, %43, %57, %44, %110] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %111, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %112 = loom.bufferize_to_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %113 = loom.subview %arg1[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %113, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %114 = loom.bufferize_to_tensor %90[%22] : memref<?xf16> -> tensor<?xf16>
                            %115 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%51 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %116 = loom.broadcast ins(%114 : tensor<?xf16>) outs(%93 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %90 : memref<?xf16>
                            %117 = loom.subview %arg2[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %89[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.broadcast ins(%118 : tensor<?xf16>) outs(%96 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %89 : memref<?xf16>
                            %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %115, %116, %119 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%87 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %127 = arith.subf %in_0, %in_1 : f16
                              %128 = math.exp %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_2 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %95 : memref<32x?xf16>
                            loom.semaphore_give %92 : memref<32x?xf16>
                            loom.semaphore_give %50 : memref<?x32xf16>
                            %121 = arith.addi %110, %55 : index
                            %122 = loom.subview %arg3[%41, %121, %42, %62] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %98[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.fill ins(%cst : f16) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %125 = linalg.matmul ins(%120, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %125 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %127 = arith.addf %in, %in_0 : f16
                              linalg.yield %127 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %83 : memref<?x?xf16>
                            scf.yield %126 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %100 = loom.alloc [1] on @L1 : memref<f16>
                          %101 = loom.semaphore_take %100 : memref<f16> -> memref<f16>
                          %102 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %102, %101 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %103 = loom.bufferize_to_tensor %101[] : memref<f16> -> tensor<f16>
                          %104 = loom.subview %arg3[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %104, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %105 = loom.bufferize_to_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %105, %103 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%82 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %110 = arith.mulf %in_0, %in_1 : f16
                            %111 = arith.addf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %101 : memref<f16>
                          loom.semaphore_give %72 : memref<?x?xf16>
                          %107 = loom.sync ins(%106 : tensor<?x?xf16>) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %81 : memref<?x?xf16>
                          %108 = loom.subview %arg7[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %109 = loom.bufferize_to_memref %107 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %109, %108 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %68 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %50 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %51 = loom.init_tensor %50[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %52 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %53 = loom.init_tensor %52[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %54 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%53 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %55 = arith.muli %43, %c256 : index
                          %56 = arith.addi %44, %55 : index
                          %57 = arith.divui %42, %c64 : index
                          %58 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %59 = loom.semaphore_take %58 : memref<?x64xf16> -> memref<?x64xf16>
                          %60 = loom.subview %arg4[%41, %56, %57, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %60, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %61 = loom.bufferize_to_tensor %59[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %62 = arith.muli %38, %21 : index
                          %63 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %64 = loom.semaphore_take %63 : memref<64x?xf16> -> memref<64x?xf16>
                          %65 = loom.subview %arg5[%41, %43, %42, 0, %62] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %66 = loom.bufferize_to_tensor %64[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %73 = loom.init_tensor %72[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %74 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %75 = linalg.matmul ins(%61, %66 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%74 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %64 : memref<64x?xf16>
                          loom.semaphore_give %59 : memref<?x64xf16>
                          %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %110 = math.exp %in_0 : f16
                            %111 = arith.mulf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          loom.semaphore_give %52 : memref<?x32xf16>
                          %77 = arith.addi %37, %c1 : index
                          %78 = arith.muli %77, %20 : index
                          %79 = arith.ceildivui %78, %22 : index
                          %80 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %81 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %82 = loom.init_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %83 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %84 = loom.init_tensor %83[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %85 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = loom.init_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %88 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %89 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %90 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %91 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %92 = loom.semaphore_take %91 : memref<32x?xf16> -> memref<32x?xf16>
                          %93 = loom.init_tensor %92[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %94 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %95 = loom.semaphore_take %94 : memref<32x?xf16> -> memref<32x?xf16>
                          %96 = loom.init_tensor %95[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %97 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = scf.for %arg18 = %c0 to %79 step %c1 iter_args(%arg19 = %76) -> (tensor<?x?xf16>) {
                            %110 = arith.muli %arg18, %22 : index
                            %111 = loom.subview %arg0[%41, %43, %57, %44, %110] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %111, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %112 = loom.bufferize_to_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %113 = loom.subview %arg1[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %113, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %114 = loom.bufferize_to_tensor %90[%22] : memref<?xf16> -> tensor<?xf16>
                            %115 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%51 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %116 = loom.broadcast ins(%114 : tensor<?xf16>) outs(%93 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %90 : memref<?xf16>
                            %117 = loom.subview %arg2[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %89[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.broadcast ins(%118 : tensor<?xf16>) outs(%96 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %89 : memref<?xf16>
                            %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %115, %116, %119 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%87 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %127 = arith.subf %in_0, %in_1 : f16
                              %128 = math.exp %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_2 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %95 : memref<32x?xf16>
                            loom.semaphore_give %92 : memref<32x?xf16>
                            loom.semaphore_give %50 : memref<?x32xf16>
                            %121 = arith.addi %110, %55 : index
                            %122 = loom.subview %arg3[%41, %121, %42, %62] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %98[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.fill ins(%cst : f16) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %125 = linalg.matmul ins(%120, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %125 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %127 = arith.addf %in, %in_0 : f16
                              linalg.yield %127 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %83 : memref<?x?xf16>
                            scf.yield %126 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %100 = loom.alloc [1] on @L1 : memref<f16>
                          %101 = loom.semaphore_take %100 : memref<f16> -> memref<f16>
                          %102 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %102, %101 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %103 = loom.bufferize_to_tensor %101[] : memref<f16> -> tensor<f16>
                          %104 = loom.subview %arg3[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %104, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %105 = loom.bufferize_to_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %105, %103 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%82 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %110 = arith.mulf %in_0, %in_1 : f16
                            %111 = arith.addf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %101 : memref<f16>
                          loom.semaphore_give %72 : memref<?x?xf16>
                          %107 = loom.sync ins(%106 : tensor<?x?xf16>) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %81 : memref<?x?xf16>
                          %108 = loom.subview %arg7[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %109 = loom.bufferize_to_memref %107 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %109, %108 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %68 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %50 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %51 = loom.init_tensor %50[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %52 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %53 = loom.init_tensor %52[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %54 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%53 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %55 = arith.muli %43, %c256 : index
                          %56 = arith.addi %44, %55 : index
                          %57 = arith.divui %42, %c64 : index
                          %58 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %59 = loom.semaphore_take %58 : memref<?x64xf16> -> memref<?x64xf16>
                          %60 = loom.subview %arg4[%41, %56, %57, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %60, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %61 = loom.bufferize_to_tensor %59[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %62 = arith.muli %38, %21 : index
                          %63 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %64 = loom.semaphore_take %63 : memref<64x?xf16> -> memref<64x?xf16>
                          %65 = loom.subview %arg5[%41, %43, %42, 0, %62] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %66 = loom.bufferize_to_tensor %64[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %73 = loom.init_tensor %72[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %74 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %75 = linalg.matmul ins(%61, %66 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%74 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %64 : memref<64x?xf16>
                          loom.semaphore_give %59 : memref<?x64xf16>
                          %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %110 = math.exp %in_0 : f16
                            %111 = arith.mulf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          loom.semaphore_give %52 : memref<?x32xf16>
                          %77 = arith.addi %37, %c1 : index
                          %78 = arith.muli %77, %20 : index
                          %79 = arith.ceildivui %78, %22 : index
                          %80 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %81 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %82 = loom.init_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %83 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %84 = loom.init_tensor %83[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %85 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = loom.init_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %88 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %89 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %90 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %91 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %92 = loom.semaphore_take %91 : memref<32x?xf16> -> memref<32x?xf16>
                          %93 = loom.init_tensor %92[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %94 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %95 = loom.semaphore_take %94 : memref<32x?xf16> -> memref<32x?xf16>
                          %96 = loom.init_tensor %95[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %97 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = scf.for %arg18 = %c0 to %79 step %c1 iter_args(%arg19 = %76) -> (tensor<?x?xf16>) {
                            %110 = arith.muli %arg18, %22 : index
                            %111 = loom.subview %arg0[%41, %43, %57, %44, %110] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %111, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %112 = loom.bufferize_to_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %113 = loom.subview %arg1[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %113, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %114 = loom.bufferize_to_tensor %90[%22] : memref<?xf16> -> tensor<?xf16>
                            %115 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%51 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %116 = loom.broadcast ins(%114 : tensor<?xf16>) outs(%93 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %90 : memref<?xf16>
                            %117 = loom.subview %arg2[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %89[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.broadcast ins(%118 : tensor<?xf16>) outs(%96 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %89 : memref<?xf16>
                            %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %115, %116, %119 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%87 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %127 = arith.subf %in_0, %in_1 : f16
                              %128 = math.exp %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_2 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %95 : memref<32x?xf16>
                            loom.semaphore_give %92 : memref<32x?xf16>
                            loom.semaphore_give %50 : memref<?x32xf16>
                            %121 = arith.addi %110, %55 : index
                            %122 = loom.subview %arg3[%41, %121, %42, %62] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %98[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.fill ins(%cst : f16) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %125 = linalg.matmul ins(%120, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %125 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %127 = arith.addf %in, %in_0 : f16
                              linalg.yield %127 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %83 : memref<?x?xf16>
                            scf.yield %126 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %100 = loom.alloc [1] on @L1 : memref<f16>
                          %101 = loom.semaphore_take %100 : memref<f16> -> memref<f16>
                          %102 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %102, %101 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %103 = loom.bufferize_to_tensor %101[] : memref<f16> -> tensor<f16>
                          %104 = loom.subview %arg3[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %104, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %105 = loom.bufferize_to_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %105, %103 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%82 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %110 = arith.mulf %in_0, %in_1 : f16
                            %111 = arith.addf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %101 : memref<f16>
                          loom.semaphore_give %72 : memref<?x?xf16>
                          %107 = loom.sync ins(%106 : tensor<?x?xf16>) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %81 : memref<?x?xf16>
                          %108 = loom.subview %arg7[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %109 = loom.bufferize_to_memref %107 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %109, %108 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %68 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %50 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %51 = loom.init_tensor %50[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %52 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %53 = loom.init_tensor %52[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %54 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%53 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %55 = arith.muli %43, %c256 : index
                          %56 = arith.addi %44, %55 : index
                          %57 = arith.divui %42, %c64 : index
                          %58 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %59 = loom.semaphore_take %58 : memref<?x64xf16> -> memref<?x64xf16>
                          %60 = loom.subview %arg4[%41, %56, %57, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %60, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %61 = loom.bufferize_to_tensor %59[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %62 = arith.muli %38, %21 : index
                          %63 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %64 = loom.semaphore_take %63 : memref<64x?xf16> -> memref<64x?xf16>
                          %65 = loom.subview %arg5[%41, %43, %42, 0, %62] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %66 = loom.bufferize_to_tensor %64[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %73 = loom.init_tensor %72[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %74 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %75 = linalg.matmul ins(%61, %66 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%74 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %64 : memref<64x?xf16>
                          loom.semaphore_give %59 : memref<?x64xf16>
                          %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %110 = math.exp %in_0 : f16
                            %111 = arith.mulf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          loom.semaphore_give %52 : memref<?x32xf16>
                          %77 = arith.addi %37, %c1 : index
                          %78 = arith.muli %77, %20 : index
                          %79 = arith.ceildivui %78, %22 : index
                          %80 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %81 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %82 = loom.init_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %83 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %84 = loom.init_tensor %83[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %85 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = loom.init_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %88 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %89 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %90 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %91 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %92 = loom.semaphore_take %91 : memref<32x?xf16> -> memref<32x?xf16>
                          %93 = loom.init_tensor %92[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %94 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %95 = loom.semaphore_take %94 : memref<32x?xf16> -> memref<32x?xf16>
                          %96 = loom.init_tensor %95[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %97 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = scf.for %arg18 = %c0 to %79 step %c1 iter_args(%arg19 = %76) -> (tensor<?x?xf16>) {
                            %110 = arith.muli %arg18, %22 : index
                            %111 = loom.subview %arg0[%41, %43, %57, %44, %110] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %111, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %112 = loom.bufferize_to_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %113 = loom.subview %arg1[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %113, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %114 = loom.bufferize_to_tensor %90[%22] : memref<?xf16> -> tensor<?xf16>
                            %115 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%51 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %116 = loom.broadcast ins(%114 : tensor<?xf16>) outs(%93 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %90 : memref<?xf16>
                            %117 = loom.subview %arg2[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %89[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.broadcast ins(%118 : tensor<?xf16>) outs(%96 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %89 : memref<?xf16>
                            %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %115, %116, %119 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%87 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %127 = arith.subf %in_0, %in_1 : f16
                              %128 = math.exp %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_2 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %95 : memref<32x?xf16>
                            loom.semaphore_give %92 : memref<32x?xf16>
                            loom.semaphore_give %50 : memref<?x32xf16>
                            %121 = arith.addi %110, %55 : index
                            %122 = loom.subview %arg3[%41, %121, %42, %62] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %98[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.fill ins(%cst : f16) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %125 = linalg.matmul ins(%120, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %125 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %127 = arith.addf %in, %in_0 : f16
                              linalg.yield %127 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %83 : memref<?x?xf16>
                            scf.yield %126 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %100 = loom.alloc [1] on @L1 : memref<f16>
                          %101 = loom.semaphore_take %100 : memref<f16> -> memref<f16>
                          %102 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %102, %101 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %103 = loom.bufferize_to_tensor %101[] : memref<f16> -> tensor<f16>
                          %104 = loom.subview %arg3[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %104, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %105 = loom.bufferize_to_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %105, %103 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%82 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %110 = arith.mulf %in_0, %in_1 : f16
                            %111 = arith.addf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %101 : memref<f16>
                          loom.semaphore_give %72 : memref<?x?xf16>
                          %107 = loom.sync ins(%106 : tensor<?x?xf16>) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %81 : memref<?x?xf16>
                          %108 = loom.subview %arg7[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %109 = loom.bufferize_to_memref %107 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %109, %108 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %68 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cst = arith.constant 0.000000e+00 : f16
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %50 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %51 = loom.init_tensor %50[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %52 = loom.semaphore_take %49 : memref<?x32xf16> -> memref<?x32xf16>
                          %53 = loom.init_tensor %52[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %54 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%53 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %55 = arith.muli %43, %c256 : index
                          %56 = arith.addi %44, %55 : index
                          %57 = arith.divui %42, %c64 : index
                          %58 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %59 = loom.semaphore_take %58 : memref<?x64xf16> -> memref<?x64xf16>
                          %60 = loom.subview %arg4[%41, %56, %57, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %60, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %61 = loom.bufferize_to_tensor %59[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %62 = arith.muli %38, %21 : index
                          %63 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %64 = loom.semaphore_take %63 : memref<64x?xf16> -> memref<64x?xf16>
                          %65 = loom.subview %arg5[%41, %43, %42, 0, %62] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %66 = loom.bufferize_to_tensor %64[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %73 = loom.init_tensor %72[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %74 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %75 = linalg.matmul ins(%61, %66 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%74 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %64 : memref<64x?xf16>
                          loom.semaphore_give %59 : memref<?x64xf16>
                          %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%73 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %110 = math.exp %in_0 : f16
                            %111 = arith.mulf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          loom.semaphore_give %52 : memref<?x32xf16>
                          %77 = arith.addi %37, %c1 : index
                          %78 = arith.muli %77, %20 : index
                          %79 = arith.ceildivui %78, %22 : index
                          %80 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %81 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %82 = loom.init_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %83 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
                          %84 = loom.init_tensor %83[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %85 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = loom.init_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %88 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %89 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %90 = loom.semaphore_take %88 : memref<?xf16> -> memref<?xf16>
                          %91 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %92 = loom.semaphore_take %91 : memref<32x?xf16> -> memref<32x?xf16>
                          %93 = loom.init_tensor %92[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %94 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %95 = loom.semaphore_take %94 : memref<32x?xf16> -> memref<32x?xf16>
                          %96 = loom.init_tensor %95[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %97 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = scf.for %arg18 = %c0 to %79 step %c1 iter_args(%arg19 = %76) -> (tensor<?x?xf16>) {
                            %110 = arith.muli %arg18, %22 : index
                            %111 = loom.subview %arg0[%41, %43, %57, %44, %110] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %111, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %112 = loom.bufferize_to_tensor %86[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %113 = loom.subview %arg1[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %113, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %114 = loom.bufferize_to_tensor %90[%22] : memref<?xf16> -> tensor<?xf16>
                            %115 = loom.broadcast ins(%48 : tensor<?xf16>) outs(%51 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %116 = loom.broadcast ins(%114 : tensor<?xf16>) outs(%93 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %90 : memref<?xf16>
                            %117 = loom.subview %arg2[%41, %42, %43, %110] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %117, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %118 = loom.bufferize_to_tensor %89[%22] : memref<?xf16> -> tensor<?xf16>
                            %119 = loom.broadcast ins(%118 : tensor<?xf16>) outs(%96 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %89 : memref<?xf16>
                            %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%112, %115, %116, %119 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%87 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %127 = arith.subf %in_0, %in_1 : f16
                              %128 = math.exp %127 : f16
                              %129 = arith.mulf %in, %128 : f16
                              %130 = arith.mulf %129, %in_2 : f16
                              linalg.yield %130 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %95 : memref<32x?xf16>
                            loom.semaphore_give %92 : memref<32x?xf16>
                            loom.semaphore_give %50 : memref<?x32xf16>
                            %121 = arith.addi %110, %55 : index
                            %122 = loom.subview %arg3[%41, %121, %42, %62] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %122, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %123 = loom.bufferize_to_tensor %98[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %124 = linalg.fill ins(%cst : f16) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %125 = linalg.matmul ins(%120, %123 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %125 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %127 = arith.addf %in, %in_0 : f16
                              linalg.yield %127 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %83 : memref<?x?xf16>
                            scf.yield %126 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %100 = loom.alloc [1] on @L1 : memref<f16>
                          %101 = loom.semaphore_take %100 : memref<f16> -> memref<f16>
                          %102 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %102, %101 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %103 = loom.bufferize_to_tensor %101[] : memref<f16> -> tensor<f16>
                          %104 = loom.subview %arg3[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %104, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %105 = loom.bufferize_to_tensor %81[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%99, %105, %103 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%82 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %110 = arith.mulf %in_0, %in_1 : f16
                            %111 = arith.addf %in, %110 : f16
                            linalg.yield %111 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %101 : memref<f16>
                          loom.semaphore_give %72 : memref<?x?xf16>
                          %107 = loom.sync ins(%106 : tensor<?x?xf16>) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %81 : memref<?x?xf16>
                          %108 = loom.subview %arg7[%41, %56, %42, %62] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %109 = loom.bufferize_to_memref %107 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %109, %108 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %68 : memref<?x?xf16>
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
