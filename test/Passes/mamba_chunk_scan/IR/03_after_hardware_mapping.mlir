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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c16, %23 : index
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %50 = loom.semaphore_take %49 : memref<?xf16> -> memref<?xf16>
                          %51 = loom.init_tensor %50[%20] : memref<?xf16> -> tensor<?xf16>
                          %52 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%48 : tensor<?xf16>) outs(%51 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %100 = arith.mulf %in, %cst_1 : f16
                            %101 = math.powf %cst, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?xf16>
                          %53 = arith.muli %43, %c256 : index
                          %54 = arith.divui %42, %c16 : index
                          %55 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %56 = loom.semaphore_take %55 : memref<?x16xf16> -> memref<?x16xf16>
                          %57 = loom.subview %arg4[%41, %53, %54, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %57, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %58 = loom.bufferize_to_tensor %56[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %59 = arith.muli %38, %21 : index
                          %60 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                          %62 = loom.subview %arg5[%41, %43, %42, %59, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %63 = loom.bufferize_to_tensor %61[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %64 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %65 = loom.semaphore_take %64 : memref<16x?xf16> -> memref<16x?xf16>
                          %66 = loom.init_tensor %65[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%63 : tensor<?x16xf16>) outs(%66 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %61 : memref<?x16xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = linalg.fill ins(%cst_0 : f16) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %73 = linalg.matmul ins(%58, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%72 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %65 : memref<16x?xf16>
                          loom.semaphore_give %56 : memref<?x16xf16>
                          %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73, %52 : tensor<?x?xf16>, tensor<?xf16>) outs(%71 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %100 = arith.mulf %in, %in_2 : f16
                            linalg.yield %100 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %68 : memref<?x?xf16>
                          loom.semaphore_give %50 : memref<?xf16>
                          %75 = arith.addi %37, %c1 : index
                          %76 = arith.muli %75, %20 : index
                          %77 = arith.ceildivui %76, %22 : index
                          %78 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %79 = loom.semaphore_take %78 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %82 = loom.semaphore_take %81 : memref<?xf16> -> memref<?xf16>
                          %83 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %84 = loom.semaphore_take %83 : memref<?xf16> -> memref<?xf16>
                          %85 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = scf.for %arg18 = %c0 to %77 step %c1 iter_args(%arg19 = %74) -> (tensor<?x?xf16>) {
                            %100 = arith.muli %arg18, %22 : index
                            %101 = loom.subview %arg0[%41, %43, %54, %44, %100] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %101, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %102 = loom.bufferize_to_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %103 = loom.subview %arg1[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %103, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %104 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                            %105 = loom.subview %arg2[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %105, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %106 = loom.bufferize_to_tensor %84[%22] : memref<?xf16> -> tensor<?xf16>
                            %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %48, %104, %106 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%80 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %111 = arith.mulf %in_3, %cst_1 : f16
                              %112 = arith.mulf %in_2, %cst_1 : f16
                              %113 = arith.subf %112, %111 : f16
                              %114 = math.powf %cst, %113 : f16
                              %115 = arith.mulf %in, %114 : f16
                              %116 = arith.mulf %115, %in_4 : f16
                              linalg.yield %116 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %84 : memref<?xf16>
                            loom.semaphore_give %82 : memref<?xf16>
                            %108 = loom.subview %arg3[%41, %53, %42, %59] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %108, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %109 = loom.bufferize_to_tensor %86[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %110 = linalg.matmul ins(%107, %109 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            loom.semaphore_give %79 : memref<?x?xf16>
                            scf.yield %110 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %88 = loom.alloc [1] on @L1 : memref<f16>
                          %89 = loom.semaphore_take %88 : memref<f16> -> memref<f16>
                          %90 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %90, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %91 = loom.bufferize_to_tensor %89[] : memref<f16> -> tensor<f16>
                          %92 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.subview %arg3[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %95, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %96 = loom.bufferize_to_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %96, %91 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%94 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %100 = arith.mulf %in_2, %in_3 : f16
                            %101 = arith.addf %in, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %89 : memref<f16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          %98 = loom.subview %arg7[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %99 = loom.bufferize_to_memref %97 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %99, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %93 : memref<?x?xf16>
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c16, %23 : index
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %50 = loom.semaphore_take %49 : memref<?xf16> -> memref<?xf16>
                          %51 = loom.init_tensor %50[%20] : memref<?xf16> -> tensor<?xf16>
                          %52 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%48 : tensor<?xf16>) outs(%51 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %100 = arith.mulf %in, %cst_1 : f16
                            %101 = math.powf %cst, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?xf16>
                          %53 = arith.muli %43, %c256 : index
                          %54 = arith.divui %42, %c16 : index
                          %55 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %56 = loom.semaphore_take %55 : memref<?x16xf16> -> memref<?x16xf16>
                          %57 = loom.subview %arg4[%41, %53, %54, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %57, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %58 = loom.bufferize_to_tensor %56[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %59 = arith.muli %38, %21 : index
                          %60 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                          %62 = loom.subview %arg5[%41, %43, %42, %59, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %63 = loom.bufferize_to_tensor %61[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %64 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %65 = loom.semaphore_take %64 : memref<16x?xf16> -> memref<16x?xf16>
                          %66 = loom.init_tensor %65[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%63 : tensor<?x16xf16>) outs(%66 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %61 : memref<?x16xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = linalg.fill ins(%cst_0 : f16) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %73 = linalg.matmul ins(%58, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%72 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %65 : memref<16x?xf16>
                          loom.semaphore_give %56 : memref<?x16xf16>
                          %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73, %52 : tensor<?x?xf16>, tensor<?xf16>) outs(%71 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %100 = arith.mulf %in, %in_2 : f16
                            linalg.yield %100 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %68 : memref<?x?xf16>
                          loom.semaphore_give %50 : memref<?xf16>
                          %75 = arith.addi %37, %c1 : index
                          %76 = arith.muli %75, %20 : index
                          %77 = arith.ceildivui %76, %22 : index
                          %78 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %79 = loom.semaphore_take %78 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %82 = loom.semaphore_take %81 : memref<?xf16> -> memref<?xf16>
                          %83 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %84 = loom.semaphore_take %83 : memref<?xf16> -> memref<?xf16>
                          %85 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = scf.for %arg18 = %c0 to %77 step %c1 iter_args(%arg19 = %74) -> (tensor<?x?xf16>) {
                            %100 = arith.muli %arg18, %22 : index
                            %101 = loom.subview %arg0[%41, %43, %54, %44, %100] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %101, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %102 = loom.bufferize_to_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %103 = loom.subview %arg1[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %103, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %104 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                            %105 = loom.subview %arg2[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %105, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %106 = loom.bufferize_to_tensor %84[%22] : memref<?xf16> -> tensor<?xf16>
                            %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %48, %104, %106 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%80 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %111 = arith.mulf %in_3, %cst_1 : f16
                              %112 = arith.mulf %in_2, %cst_1 : f16
                              %113 = arith.subf %112, %111 : f16
                              %114 = math.powf %cst, %113 : f16
                              %115 = arith.mulf %in, %114 : f16
                              %116 = arith.mulf %115, %in_4 : f16
                              linalg.yield %116 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %84 : memref<?xf16>
                            loom.semaphore_give %82 : memref<?xf16>
                            %108 = loom.subview %arg3[%41, %53, %42, %59] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %108, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %109 = loom.bufferize_to_tensor %86[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %110 = linalg.matmul ins(%107, %109 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            loom.semaphore_give %79 : memref<?x?xf16>
                            scf.yield %110 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %88 = loom.alloc [1] on @L1 : memref<f16>
                          %89 = loom.semaphore_take %88 : memref<f16> -> memref<f16>
                          %90 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %90, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %91 = loom.bufferize_to_tensor %89[] : memref<f16> -> tensor<f16>
                          %92 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.subview %arg3[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %95, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %96 = loom.bufferize_to_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %96, %91 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%94 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %100 = arith.mulf %in_2, %in_3 : f16
                            %101 = arith.addf %in, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %89 : memref<f16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          %98 = loom.subview %arg7[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %99 = loom.bufferize_to_memref %97 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %99, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %93 : memref<?x?xf16>
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c16, %23 : index
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %50 = loom.semaphore_take %49 : memref<?xf16> -> memref<?xf16>
                          %51 = loom.init_tensor %50[%20] : memref<?xf16> -> tensor<?xf16>
                          %52 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%48 : tensor<?xf16>) outs(%51 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %100 = arith.mulf %in, %cst_1 : f16
                            %101 = math.powf %cst, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?xf16>
                          %53 = arith.muli %43, %c256 : index
                          %54 = arith.divui %42, %c16 : index
                          %55 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %56 = loom.semaphore_take %55 : memref<?x16xf16> -> memref<?x16xf16>
                          %57 = loom.subview %arg4[%41, %53, %54, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %57, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %58 = loom.bufferize_to_tensor %56[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %59 = arith.muli %38, %21 : index
                          %60 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                          %62 = loom.subview %arg5[%41, %43, %42, %59, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %63 = loom.bufferize_to_tensor %61[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %64 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %65 = loom.semaphore_take %64 : memref<16x?xf16> -> memref<16x?xf16>
                          %66 = loom.init_tensor %65[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%63 : tensor<?x16xf16>) outs(%66 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %61 : memref<?x16xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = linalg.fill ins(%cst_0 : f16) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %73 = linalg.matmul ins(%58, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%72 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %65 : memref<16x?xf16>
                          loom.semaphore_give %56 : memref<?x16xf16>
                          %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73, %52 : tensor<?x?xf16>, tensor<?xf16>) outs(%71 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %100 = arith.mulf %in, %in_2 : f16
                            linalg.yield %100 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %68 : memref<?x?xf16>
                          loom.semaphore_give %50 : memref<?xf16>
                          %75 = arith.addi %37, %c1 : index
                          %76 = arith.muli %75, %20 : index
                          %77 = arith.ceildivui %76, %22 : index
                          %78 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %79 = loom.semaphore_take %78 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %82 = loom.semaphore_take %81 : memref<?xf16> -> memref<?xf16>
                          %83 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %84 = loom.semaphore_take %83 : memref<?xf16> -> memref<?xf16>
                          %85 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = scf.for %arg18 = %c0 to %77 step %c1 iter_args(%arg19 = %74) -> (tensor<?x?xf16>) {
                            %100 = arith.muli %arg18, %22 : index
                            %101 = loom.subview %arg0[%41, %43, %54, %44, %100] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %101, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %102 = loom.bufferize_to_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %103 = loom.subview %arg1[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %103, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %104 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                            %105 = loom.subview %arg2[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %105, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %106 = loom.bufferize_to_tensor %84[%22] : memref<?xf16> -> tensor<?xf16>
                            %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %48, %104, %106 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%80 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %111 = arith.mulf %in_3, %cst_1 : f16
                              %112 = arith.mulf %in_2, %cst_1 : f16
                              %113 = arith.subf %112, %111 : f16
                              %114 = math.powf %cst, %113 : f16
                              %115 = arith.mulf %in, %114 : f16
                              %116 = arith.mulf %115, %in_4 : f16
                              linalg.yield %116 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %84 : memref<?xf16>
                            loom.semaphore_give %82 : memref<?xf16>
                            %108 = loom.subview %arg3[%41, %53, %42, %59] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %108, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %109 = loom.bufferize_to_tensor %86[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %110 = linalg.matmul ins(%107, %109 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            loom.semaphore_give %79 : memref<?x?xf16>
                            scf.yield %110 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %88 = loom.alloc [1] on @L1 : memref<f16>
                          %89 = loom.semaphore_take %88 : memref<f16> -> memref<f16>
                          %90 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %90, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %91 = loom.bufferize_to_tensor %89[] : memref<f16> -> tensor<f16>
                          %92 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.subview %arg3[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %95, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %96 = loom.bufferize_to_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %96, %91 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%94 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %100 = arith.mulf %in_2, %in_3 : f16
                            %101 = arith.addf %in, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %89 : memref<f16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          %98 = loom.subview %arg7[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %99 = loom.bufferize_to_memref %97 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %99, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %93 : memref<?x?xf16>
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c16, %23 : index
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %50 = loom.semaphore_take %49 : memref<?xf16> -> memref<?xf16>
                          %51 = loom.init_tensor %50[%20] : memref<?xf16> -> tensor<?xf16>
                          %52 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%48 : tensor<?xf16>) outs(%51 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %100 = arith.mulf %in, %cst_1 : f16
                            %101 = math.powf %cst, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?xf16>
                          %53 = arith.muli %43, %c256 : index
                          %54 = arith.divui %42, %c16 : index
                          %55 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %56 = loom.semaphore_take %55 : memref<?x16xf16> -> memref<?x16xf16>
                          %57 = loom.subview %arg4[%41, %53, %54, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %57, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %58 = loom.bufferize_to_tensor %56[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %59 = arith.muli %38, %21 : index
                          %60 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                          %62 = loom.subview %arg5[%41, %43, %42, %59, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %63 = loom.bufferize_to_tensor %61[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %64 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %65 = loom.semaphore_take %64 : memref<16x?xf16> -> memref<16x?xf16>
                          %66 = loom.init_tensor %65[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%63 : tensor<?x16xf16>) outs(%66 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %61 : memref<?x16xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = linalg.fill ins(%cst_0 : f16) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %73 = linalg.matmul ins(%58, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%72 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %65 : memref<16x?xf16>
                          loom.semaphore_give %56 : memref<?x16xf16>
                          %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73, %52 : tensor<?x?xf16>, tensor<?xf16>) outs(%71 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %100 = arith.mulf %in, %in_2 : f16
                            linalg.yield %100 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %68 : memref<?x?xf16>
                          loom.semaphore_give %50 : memref<?xf16>
                          %75 = arith.addi %37, %c1 : index
                          %76 = arith.muli %75, %20 : index
                          %77 = arith.ceildivui %76, %22 : index
                          %78 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %79 = loom.semaphore_take %78 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %82 = loom.semaphore_take %81 : memref<?xf16> -> memref<?xf16>
                          %83 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %84 = loom.semaphore_take %83 : memref<?xf16> -> memref<?xf16>
                          %85 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = scf.for %arg18 = %c0 to %77 step %c1 iter_args(%arg19 = %74) -> (tensor<?x?xf16>) {
                            %100 = arith.muli %arg18, %22 : index
                            %101 = loom.subview %arg0[%41, %43, %54, %44, %100] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %101, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %102 = loom.bufferize_to_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %103 = loom.subview %arg1[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %103, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %104 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                            %105 = loom.subview %arg2[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %105, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %106 = loom.bufferize_to_tensor %84[%22] : memref<?xf16> -> tensor<?xf16>
                            %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %48, %104, %106 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%80 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %111 = arith.mulf %in_3, %cst_1 : f16
                              %112 = arith.mulf %in_2, %cst_1 : f16
                              %113 = arith.subf %112, %111 : f16
                              %114 = math.powf %cst, %113 : f16
                              %115 = arith.mulf %in, %114 : f16
                              %116 = arith.mulf %115, %in_4 : f16
                              linalg.yield %116 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %84 : memref<?xf16>
                            loom.semaphore_give %82 : memref<?xf16>
                            %108 = loom.subview %arg3[%41, %53, %42, %59] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %108, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %109 = loom.bufferize_to_tensor %86[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %110 = linalg.matmul ins(%107, %109 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            loom.semaphore_give %79 : memref<?x?xf16>
                            scf.yield %110 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %88 = loom.alloc [1] on @L1 : memref<f16>
                          %89 = loom.semaphore_take %88 : memref<f16> -> memref<f16>
                          %90 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %90, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %91 = loom.bufferize_to_tensor %89[] : memref<f16> -> tensor<f16>
                          %92 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.subview %arg3[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %95, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %96 = loom.bufferize_to_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %96, %91 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%94 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %100 = arith.mulf %in_2, %in_3 : f16
                            %101 = arith.addf %in, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %89 : memref<f16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          %98 = loom.subview %arg7[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %99 = loom.bufferize_to_memref %97 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %99, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %93 : memref<?x?xf16>
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c16, %23 : index
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %50 = loom.semaphore_take %49 : memref<?xf16> -> memref<?xf16>
                          %51 = loom.init_tensor %50[%20] : memref<?xf16> -> tensor<?xf16>
                          %52 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%48 : tensor<?xf16>) outs(%51 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %100 = arith.mulf %in, %cst_1 : f16
                            %101 = math.powf %cst, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?xf16>
                          %53 = arith.muli %43, %c256 : index
                          %54 = arith.divui %42, %c16 : index
                          %55 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %56 = loom.semaphore_take %55 : memref<?x16xf16> -> memref<?x16xf16>
                          %57 = loom.subview %arg4[%41, %53, %54, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %57, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %58 = loom.bufferize_to_tensor %56[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %59 = arith.muli %38, %21 : index
                          %60 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                          %62 = loom.subview %arg5[%41, %43, %42, %59, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %63 = loom.bufferize_to_tensor %61[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %64 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %65 = loom.semaphore_take %64 : memref<16x?xf16> -> memref<16x?xf16>
                          %66 = loom.init_tensor %65[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%63 : tensor<?x16xf16>) outs(%66 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %61 : memref<?x16xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = linalg.fill ins(%cst_0 : f16) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %73 = linalg.matmul ins(%58, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%72 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %65 : memref<16x?xf16>
                          loom.semaphore_give %56 : memref<?x16xf16>
                          %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73, %52 : tensor<?x?xf16>, tensor<?xf16>) outs(%71 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %100 = arith.mulf %in, %in_2 : f16
                            linalg.yield %100 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %68 : memref<?x?xf16>
                          loom.semaphore_give %50 : memref<?xf16>
                          %75 = arith.addi %37, %c1 : index
                          %76 = arith.muli %75, %20 : index
                          %77 = arith.ceildivui %76, %22 : index
                          %78 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %79 = loom.semaphore_take %78 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %82 = loom.semaphore_take %81 : memref<?xf16> -> memref<?xf16>
                          %83 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %84 = loom.semaphore_take %83 : memref<?xf16> -> memref<?xf16>
                          %85 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = scf.for %arg18 = %c0 to %77 step %c1 iter_args(%arg19 = %74) -> (tensor<?x?xf16>) {
                            %100 = arith.muli %arg18, %22 : index
                            %101 = loom.subview %arg0[%41, %43, %54, %44, %100] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %101, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %102 = loom.bufferize_to_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %103 = loom.subview %arg1[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %103, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %104 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                            %105 = loom.subview %arg2[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %105, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %106 = loom.bufferize_to_tensor %84[%22] : memref<?xf16> -> tensor<?xf16>
                            %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %48, %104, %106 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%80 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %111 = arith.mulf %in_3, %cst_1 : f16
                              %112 = arith.mulf %in_2, %cst_1 : f16
                              %113 = arith.subf %112, %111 : f16
                              %114 = math.powf %cst, %113 : f16
                              %115 = arith.mulf %in, %114 : f16
                              %116 = arith.mulf %115, %in_4 : f16
                              linalg.yield %116 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %84 : memref<?xf16>
                            loom.semaphore_give %82 : memref<?xf16>
                            %108 = loom.subview %arg3[%41, %53, %42, %59] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %108, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %109 = loom.bufferize_to_tensor %86[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %110 = linalg.matmul ins(%107, %109 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            loom.semaphore_give %79 : memref<?x?xf16>
                            scf.yield %110 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %88 = loom.alloc [1] on @L1 : memref<f16>
                          %89 = loom.semaphore_take %88 : memref<f16> -> memref<f16>
                          %90 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %90, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %91 = loom.bufferize_to_tensor %89[] : memref<f16> -> tensor<f16>
                          %92 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.subview %arg3[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %95, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %96 = loom.bufferize_to_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %96, %91 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%94 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %100 = arith.mulf %in_2, %in_3 : f16
                            %101 = arith.addf %in, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %89 : memref<f16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          %98 = loom.subview %arg7[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %99 = loom.bufferize_to_memref %97 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %99, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %93 : memref<?x?xf16>
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c16, %23 : index
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %50 = loom.semaphore_take %49 : memref<?xf16> -> memref<?xf16>
                          %51 = loom.init_tensor %50[%20] : memref<?xf16> -> tensor<?xf16>
                          %52 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%48 : tensor<?xf16>) outs(%51 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %100 = arith.mulf %in, %cst_1 : f16
                            %101 = math.powf %cst, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?xf16>
                          %53 = arith.muli %43, %c256 : index
                          %54 = arith.divui %42, %c16 : index
                          %55 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %56 = loom.semaphore_take %55 : memref<?x16xf16> -> memref<?x16xf16>
                          %57 = loom.subview %arg4[%41, %53, %54, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %57, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %58 = loom.bufferize_to_tensor %56[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %59 = arith.muli %38, %21 : index
                          %60 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                          %62 = loom.subview %arg5[%41, %43, %42, %59, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %63 = loom.bufferize_to_tensor %61[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %64 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %65 = loom.semaphore_take %64 : memref<16x?xf16> -> memref<16x?xf16>
                          %66 = loom.init_tensor %65[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%63 : tensor<?x16xf16>) outs(%66 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %61 : memref<?x16xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = linalg.fill ins(%cst_0 : f16) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %73 = linalg.matmul ins(%58, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%72 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %65 : memref<16x?xf16>
                          loom.semaphore_give %56 : memref<?x16xf16>
                          %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73, %52 : tensor<?x?xf16>, tensor<?xf16>) outs(%71 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %100 = arith.mulf %in, %in_2 : f16
                            linalg.yield %100 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %68 : memref<?x?xf16>
                          loom.semaphore_give %50 : memref<?xf16>
                          %75 = arith.addi %37, %c1 : index
                          %76 = arith.muli %75, %20 : index
                          %77 = arith.ceildivui %76, %22 : index
                          %78 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %79 = loom.semaphore_take %78 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %82 = loom.semaphore_take %81 : memref<?xf16> -> memref<?xf16>
                          %83 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %84 = loom.semaphore_take %83 : memref<?xf16> -> memref<?xf16>
                          %85 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = scf.for %arg18 = %c0 to %77 step %c1 iter_args(%arg19 = %74) -> (tensor<?x?xf16>) {
                            %100 = arith.muli %arg18, %22 : index
                            %101 = loom.subview %arg0[%41, %43, %54, %44, %100] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %101, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %102 = loom.bufferize_to_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %103 = loom.subview %arg1[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %103, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %104 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                            %105 = loom.subview %arg2[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %105, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %106 = loom.bufferize_to_tensor %84[%22] : memref<?xf16> -> tensor<?xf16>
                            %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %48, %104, %106 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%80 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %111 = arith.mulf %in_3, %cst_1 : f16
                              %112 = arith.mulf %in_2, %cst_1 : f16
                              %113 = arith.subf %112, %111 : f16
                              %114 = math.powf %cst, %113 : f16
                              %115 = arith.mulf %in, %114 : f16
                              %116 = arith.mulf %115, %in_4 : f16
                              linalg.yield %116 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %84 : memref<?xf16>
                            loom.semaphore_give %82 : memref<?xf16>
                            %108 = loom.subview %arg3[%41, %53, %42, %59] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %108, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %109 = loom.bufferize_to_tensor %86[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %110 = linalg.matmul ins(%107, %109 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            loom.semaphore_give %79 : memref<?x?xf16>
                            scf.yield %110 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %88 = loom.alloc [1] on @L1 : memref<f16>
                          %89 = loom.semaphore_take %88 : memref<f16> -> memref<f16>
                          %90 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %90, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %91 = loom.bufferize_to_tensor %89[] : memref<f16> -> tensor<f16>
                          %92 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.subview %arg3[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %95, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %96 = loom.bufferize_to_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %96, %91 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%94 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %100 = arith.mulf %in_2, %in_3 : f16
                            %101 = arith.addf %in, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %89 : memref<f16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          %98 = loom.subview %arg7[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %99 = loom.bufferize_to_memref %97 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %99, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %93 : memref<?x?xf16>
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c16, %23 : index
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %50 = loom.semaphore_take %49 : memref<?xf16> -> memref<?xf16>
                          %51 = loom.init_tensor %50[%20] : memref<?xf16> -> tensor<?xf16>
                          %52 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%48 : tensor<?xf16>) outs(%51 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %100 = arith.mulf %in, %cst_1 : f16
                            %101 = math.powf %cst, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?xf16>
                          %53 = arith.muli %43, %c256 : index
                          %54 = arith.divui %42, %c16 : index
                          %55 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %56 = loom.semaphore_take %55 : memref<?x16xf16> -> memref<?x16xf16>
                          %57 = loom.subview %arg4[%41, %53, %54, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %57, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %58 = loom.bufferize_to_tensor %56[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %59 = arith.muli %38, %21 : index
                          %60 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                          %62 = loom.subview %arg5[%41, %43, %42, %59, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %63 = loom.bufferize_to_tensor %61[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %64 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %65 = loom.semaphore_take %64 : memref<16x?xf16> -> memref<16x?xf16>
                          %66 = loom.init_tensor %65[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%63 : tensor<?x16xf16>) outs(%66 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %61 : memref<?x16xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = linalg.fill ins(%cst_0 : f16) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %73 = linalg.matmul ins(%58, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%72 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %65 : memref<16x?xf16>
                          loom.semaphore_give %56 : memref<?x16xf16>
                          %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73, %52 : tensor<?x?xf16>, tensor<?xf16>) outs(%71 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %100 = arith.mulf %in, %in_2 : f16
                            linalg.yield %100 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %68 : memref<?x?xf16>
                          loom.semaphore_give %50 : memref<?xf16>
                          %75 = arith.addi %37, %c1 : index
                          %76 = arith.muli %75, %20 : index
                          %77 = arith.ceildivui %76, %22 : index
                          %78 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %79 = loom.semaphore_take %78 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %82 = loom.semaphore_take %81 : memref<?xf16> -> memref<?xf16>
                          %83 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %84 = loom.semaphore_take %83 : memref<?xf16> -> memref<?xf16>
                          %85 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = scf.for %arg18 = %c0 to %77 step %c1 iter_args(%arg19 = %74) -> (tensor<?x?xf16>) {
                            %100 = arith.muli %arg18, %22 : index
                            %101 = loom.subview %arg0[%41, %43, %54, %44, %100] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %101, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %102 = loom.bufferize_to_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %103 = loom.subview %arg1[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %103, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %104 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                            %105 = loom.subview %arg2[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %105, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %106 = loom.bufferize_to_tensor %84[%22] : memref<?xf16> -> tensor<?xf16>
                            %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %48, %104, %106 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%80 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %111 = arith.mulf %in_3, %cst_1 : f16
                              %112 = arith.mulf %in_2, %cst_1 : f16
                              %113 = arith.subf %112, %111 : f16
                              %114 = math.powf %cst, %113 : f16
                              %115 = arith.mulf %in, %114 : f16
                              %116 = arith.mulf %115, %in_4 : f16
                              linalg.yield %116 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %84 : memref<?xf16>
                            loom.semaphore_give %82 : memref<?xf16>
                            %108 = loom.subview %arg3[%41, %53, %42, %59] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %108, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %109 = loom.bufferize_to_tensor %86[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %110 = linalg.matmul ins(%107, %109 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            loom.semaphore_give %79 : memref<?x?xf16>
                            scf.yield %110 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %88 = loom.alloc [1] on @L1 : memref<f16>
                          %89 = loom.semaphore_take %88 : memref<f16> -> memref<f16>
                          %90 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %90, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %91 = loom.bufferize_to_tensor %89[] : memref<f16> -> tensor<f16>
                          %92 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.subview %arg3[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %95, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %96 = loom.bufferize_to_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %96, %91 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%94 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %100 = arith.mulf %in_2, %in_3 : f16
                            %101 = arith.addf %in, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %89 : memref<f16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          %98 = loom.subview %arg7[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %99 = loom.bufferize_to_memref %97 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %99, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %93 : memref<?x?xf16>
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.442380e+00 : f16
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_b {upper_bound = 2 : index} : index
      %25 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %26 = arith.ceildivui %c16, %23 : index
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %48 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.alloc [%20] on @L1 : memref<?xf16>
                          %50 = loom.semaphore_take %49 : memref<?xf16> -> memref<?xf16>
                          %51 = loom.init_tensor %50[%20] : memref<?xf16> -> tensor<?xf16>
                          %52 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%48 : tensor<?xf16>) outs(%51 : tensor<?xf16>) {
                          ^bb0(%in: f16, %out: f16):
                            %100 = arith.mulf %in, %cst_1 : f16
                            %101 = math.powf %cst, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?xf16>
                          %53 = arith.muli %43, %c256 : index
                          %54 = arith.divui %42, %c16 : index
                          %55 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %56 = loom.semaphore_take %55 : memref<?x16xf16> -> memref<?x16xf16>
                          %57 = loom.subview %arg4[%41, %53, %54, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %57, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %58 = loom.bufferize_to_tensor %56[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %59 = arith.muli %38, %21 : index
                          %60 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                          %62 = loom.subview %arg5[%41, %43, %42, %59, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %63 = loom.bufferize_to_tensor %61[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %64 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %65 = loom.semaphore_take %64 : memref<16x?xf16> -> memref<16x?xf16>
                          %66 = loom.init_tensor %65[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%63 : tensor<?x16xf16>) outs(%66 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %61 : memref<?x16xf16>
                          %67 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %68 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %69 = loom.init_tensor %68[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %70 = loom.semaphore_take %67 : memref<?x?xf16> -> memref<?x?xf16>
                          %71 = loom.init_tensor %70[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %72 = linalg.fill ins(%cst_0 : f16) outs(%69 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %73 = linalg.matmul ins(%58, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%72 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %65 : memref<16x?xf16>
                          loom.semaphore_give %56 : memref<?x16xf16>
                          %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73, %52 : tensor<?x?xf16>, tensor<?xf16>) outs(%71 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %out: f16):
                            %100 = arith.mulf %in, %in_2 : f16
                            linalg.yield %100 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %68 : memref<?x?xf16>
                          loom.semaphore_give %50 : memref<?xf16>
                          %75 = arith.addi %37, %c1 : index
                          %76 = arith.muli %75, %20 : index
                          %77 = arith.ceildivui %76, %22 : index
                          %78 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %79 = loom.semaphore_take %78 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %82 = loom.semaphore_take %81 : memref<?xf16> -> memref<?xf16>
                          %83 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %84 = loom.semaphore_take %83 : memref<?xf16> -> memref<?xf16>
                          %85 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %86 = loom.semaphore_take %85 : memref<?x?xf16> -> memref<?x?xf16>
                          %87 = scf.for %arg18 = %c0 to %77 step %c1 iter_args(%arg19 = %74) -> (tensor<?x?xf16>) {
                            %100 = arith.muli %arg18, %22 : index
                            %101 = loom.subview %arg0[%41, %43, %54, %44, %100] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %101, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %102 = loom.bufferize_to_tensor %79[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %103 = loom.subview %arg1[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %103, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %104 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                            %105 = loom.subview %arg2[%41, %42, %43, %100] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %105, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %106 = loom.bufferize_to_tensor %84[%22] : memref<?xf16> -> tensor<?xf16>
                            %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%102, %48, %104, %106 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%80 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
                              %111 = arith.mulf %in_3, %cst_1 : f16
                              %112 = arith.mulf %in_2, %cst_1 : f16
                              %113 = arith.subf %112, %111 : f16
                              %114 = math.powf %cst, %113 : f16
                              %115 = arith.mulf %in, %114 : f16
                              %116 = arith.mulf %115, %in_4 : f16
                              linalg.yield %116 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %84 : memref<?xf16>
                            loom.semaphore_give %82 : memref<?xf16>
                            %108 = loom.subview %arg3[%41, %53, %42, %59] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %108, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %109 = loom.bufferize_to_tensor %86[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %110 = linalg.matmul ins(%107, %109 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %86 : memref<?x?xf16>
                            loom.semaphore_give %79 : memref<?x?xf16>
                            scf.yield %110 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %46 : memref<?xf16>
                          %88 = loom.alloc [1] on @L1 : memref<f16>
                          %89 = loom.semaphore_take %88 : memref<f16> -> memref<f16>
                          %90 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %90, %89 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %91 = loom.bufferize_to_tensor %89[] : memref<f16> -> tensor<f16>
                          %92 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.subview %arg3[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %95, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %96 = loom.bufferize_to_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %96, %91 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%94 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
                            %100 = arith.mulf %in_2, %in_3 : f16
                            %101 = arith.addf %in, %100 : f16
                            linalg.yield %101 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %89 : memref<f16>
                          loom.semaphore_give %70 : memref<?x?xf16>
                          %98 = loom.subview %arg7[%41, %53, %42, %59] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %99 = loom.bufferize_to_memref %97 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %99, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %93 : memref<?x?xf16>
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
