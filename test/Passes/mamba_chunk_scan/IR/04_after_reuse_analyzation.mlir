module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
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
  module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %25 = arith.ceildivui %c16, %23 : index
      %26 = arith.ceildivui %c256, %20 : index
      %27 = arith.ceildivui %c64, %21 : index
      %28 = arith.ceildivui %c8, %24 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                %29 = arith.ceildivui %25, %c2 : index
                scf.for %arg13 = %c0 to %29 step %c1 {
                  %30 = arith.ceildivui %26, %c2 : index
                  scf.for %arg14 = %c0 to %30 step %c1 {
                    %31 = arith.ceildivui %27, %c2 : index
                    scf.for %arg15 = %c0 to %31 step %c1 {
                      %32 = arith.ceildivui %28, %c2 : index
                      scf.for %arg16 = %c0 to %32 step %c1 {
                        %33 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                        %34 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                        %35 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                        %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg16)
                        %37 = arith.muli %33, %23 : index
                        %38 = arith.muli %36, %24 : index
                        %39 = arith.muli %34, %20 : index
                        %40 = loom.alloc [%20] on @L1 : memref<?xf16>
                        %41 = loom.semaphore_take %40 : memref<?xf16> -> memref<?xf16>
                        %42 = loom.subview %arg1[%arg11, %37, %38, %39] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %43 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %44 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %45 = loom.semaphore_take %44 : memref<?xf32> -> memref<?xf32>
                        %46 = loom.init_tensor %45[%20] : memref<?xf32> -> tensor<?xf32>
                        %47 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%46 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %104 = arith.extf %in : f16 to f32
                          linalg.yield %104 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %48 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %49 = loom.semaphore_take %48 : memref<?xf32> -> memref<?xf32>
                        %50 = loom.init_tensor %49[%20] : memref<?xf32> -> tensor<?xf32>
                        %51 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%47 : tensor<?xf32>) outs(%50 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %104 = arith.truncf %cst_0 : f64 to f32
                          %105 = arith.mulf %in, %104 : f32
                          %106 = math.powf %cst, %105 : f32
                          linalg.yield %106 : f32
                        } -> tensor<?xf32>
                        %52 = arith.muli %38, %c256 : index
                        %53 = arith.divui %37, %c16 : index
                        %54 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %55 = loom.semaphore_take %54 : memref<?x16xf16> -> memref<?x16xf16>
                        %56 = loom.subview %arg4[%arg11, %52, %53, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %56, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %57 = loom.bufferize_to_tensor %55[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %58 = arith.muli %35, %21 : index
                        %59 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %60 = loom.semaphore_take %59 : memref<?x16xf16> -> memref<?x16xf16>
                        %61 = loom.subview %arg5[%arg11, %38, %37, %58, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %61, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %62 = loom.bufferize_to_tensor %60[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %63 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %64 = loom.semaphore_take %63 : memref<16x?xf16> -> memref<16x?xf16>
                        %65 = loom.init_tensor %64[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%62 : tensor<?x16xf16>) outs(%65 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %60 : memref<?x16xf16>
                        %66 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %67 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %68 = loom.init_tensor %67[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %69 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %70 = loom.init_tensor %69[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %71 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %72 = linalg.matmul ins(%57, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%71 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %64 : memref<16x?xf16>
                        loom.semaphore_give %55 : memref<?x16xf16>
                        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %51 : tensor<?x?xf32>, tensor<?xf32>) outs(%70 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %104 = arith.mulf %in, %in_2 : f32
                          linalg.yield %104 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %67 : memref<?x?xf32>
                        loom.semaphore_give %49 : memref<?xf32>
                        %74 = arith.addi %34, %c1 : index
                        %75 = arith.muli %74, %20 : index
                        %76 = arith.ceildivui %75, %22 : index
                        %77 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                        %79 = loom.init_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %80 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %81 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %82 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %83 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %84 = loom.semaphore_take %83 : memref<?xf32> -> memref<?xf32>
                        %85 = loom.init_tensor %84[%22] : memref<?xf32> -> tensor<?xf32>
                        %86 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %87 = loom.semaphore_take %86 : memref<?xf32> -> memref<?xf32>
                        %88 = loom.init_tensor %87[%22] : memref<?xf32> -> tensor<?xf32>
                        %89 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %90 = loom.semaphore_take %89 : memref<?x?xf16> -> memref<?x?xf16>
                        %91 = scf.for %arg17 = %c0 to %76 step %c1 iter_args(%arg18 = %73) -> (tensor<?x?xf32>) {
                          %104 = arith.muli %arg17, %22 : index
                          %105 = loom.subview %arg0[%arg11, %38, %53, %39, %104] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %105, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %106 = loom.bufferize_to_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %107 = loom.subview %arg1[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %107, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %108 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                          %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%108 : tensor<?xf16>) outs(%85 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %82 : memref<?xf16>
                          %110 = loom.subview %arg2[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %110, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %111 = loom.bufferize_to_tensor %81[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%111 : tensor<?xf16>) outs(%88 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %81 : memref<?xf16>
                          %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %47, %109, %112 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%79 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %117 = arith.truncf %cst_0 : f64 to f32
                            %118 = arith.mulf %in_3, %117 : f32
                            %119 = arith.mulf %in_2, %117 : f32
                            %120 = arith.subf %119, %118 : f32
                            %121 = math.powf %cst, %120 : f32
                            %122 = arith.extf %in : f16 to f32
                            %123 = arith.mulf %122, %121 : f32
                            %124 = arith.mulf %123, %in_4 : f32
                            %125 = arith.truncf %124 : f32 to f16
                            linalg.yield %125 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?xf32>
                          loom.semaphore_give %84 : memref<?xf32>
                          %114 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %114, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %115 = loom.bufferize_to_tensor %90[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %116 = linalg.matmul ins(%113, %115 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          scf.yield %116 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %45 : memref<?xf32>
                        %92 = loom.alloc [1] on @L1 : memref<f16>
                        %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
                        %94 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        loom.copy %94, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %95 = loom.bufferize_to_tensor %93[] : memref<f16> -> tensor<f16>
                        %96 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                        %98 = loom.init_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %99 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %99, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %100 = loom.bufferize_to_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%91, %100, %95 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%98 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %104 = arith.extf %in_3 : f16 to f32
                          %105 = arith.extf %in_2 : f16 to f32
                          %106 = arith.mulf %105, %104 : f32
                          %107 = arith.addf %in, %106 : f32
                          %108 = arith.truncf %107 : f32 to f16
                          linalg.yield %108 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %93 : memref<f16>
                        loom.semaphore_give %69 : memref<?x?xf32>
                        %102 = loom.subview %arg7[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %103 = loom.bufferize_to_memref %101 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %103, %102 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %97 : memref<?x?xf16>
                      } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %25 = arith.ceildivui %c16, %23 : index
      %26 = arith.ceildivui %c256, %20 : index
      %27 = arith.ceildivui %c64, %21 : index
      %28 = arith.ceildivui %c8, %24 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                %29 = arith.ceildivui %25, %c2 : index
                scf.for %arg13 = %c0 to %29 step %c1 {
                  %30 = arith.ceildivui %26, %c2 : index
                  scf.for %arg14 = %c0 to %30 step %c1 {
                    %31 = arith.ceildivui %27, %c2 : index
                    scf.for %arg15 = %c0 to %31 step %c1 {
                      %32 = arith.ceildivui %28, %c4 : index
                      scf.for %arg16 = %c0 to %32 step %c1 {
                        %33 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                        %34 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                        %35 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                        %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg12, %arg16)
                        %37 = arith.muli %33, %23 : index
                        %38 = arith.muli %36, %24 : index
                        %39 = arith.muli %34, %20 : index
                        %40 = loom.alloc [%20] on @L1 : memref<?xf16>
                        %41 = loom.semaphore_take %40 : memref<?xf16> -> memref<?xf16>
                        %42 = loom.subview %arg1[%arg11, %37, %38, %39] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %43 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %44 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %45 = loom.semaphore_take %44 : memref<?xf32> -> memref<?xf32>
                        %46 = loom.init_tensor %45[%20] : memref<?xf32> -> tensor<?xf32>
                        %47 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%46 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %104 = arith.extf %in : f16 to f32
                          linalg.yield %104 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %48 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %49 = loom.semaphore_take %48 : memref<?xf32> -> memref<?xf32>
                        %50 = loom.init_tensor %49[%20] : memref<?xf32> -> tensor<?xf32>
                        %51 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%47 : tensor<?xf32>) outs(%50 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %104 = arith.truncf %cst_0 : f64 to f32
                          %105 = arith.mulf %in, %104 : f32
                          %106 = math.powf %cst, %105 : f32
                          linalg.yield %106 : f32
                        } -> tensor<?xf32>
                        %52 = arith.muli %38, %c256 : index
                        %53 = arith.divui %37, %c16 : index
                        %54 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %55 = loom.semaphore_take %54 : memref<?x16xf16> -> memref<?x16xf16>
                        %56 = loom.subview %arg4[%arg11, %52, %53, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %56, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %57 = loom.bufferize_to_tensor %55[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %58 = arith.muli %35, %21 : index
                        %59 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %60 = loom.semaphore_take %59 : memref<?x16xf16> -> memref<?x16xf16>
                        %61 = loom.subview %arg5[%arg11, %38, %37, %58, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %61, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %62 = loom.bufferize_to_tensor %60[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %63 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %64 = loom.semaphore_take %63 : memref<16x?xf16> -> memref<16x?xf16>
                        %65 = loom.init_tensor %64[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%62 : tensor<?x16xf16>) outs(%65 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %60 : memref<?x16xf16>
                        %66 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %67 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %68 = loom.init_tensor %67[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %69 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %70 = loom.init_tensor %69[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %71 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %72 = linalg.matmul ins(%57, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%71 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %64 : memref<16x?xf16>
                        loom.semaphore_give %55 : memref<?x16xf16>
                        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %51 : tensor<?x?xf32>, tensor<?xf32>) outs(%70 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %104 = arith.mulf %in, %in_2 : f32
                          linalg.yield %104 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %67 : memref<?x?xf32>
                        loom.semaphore_give %49 : memref<?xf32>
                        %74 = arith.addi %34, %c1 : index
                        %75 = arith.muli %74, %20 : index
                        %76 = arith.ceildivui %75, %22 : index
                        %77 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                        %79 = loom.init_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %80 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %81 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %82 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %83 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %84 = loom.semaphore_take %83 : memref<?xf32> -> memref<?xf32>
                        %85 = loom.init_tensor %84[%22] : memref<?xf32> -> tensor<?xf32>
                        %86 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %87 = loom.semaphore_take %86 : memref<?xf32> -> memref<?xf32>
                        %88 = loom.init_tensor %87[%22] : memref<?xf32> -> tensor<?xf32>
                        %89 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %90 = loom.semaphore_take %89 : memref<?x?xf16> -> memref<?x?xf16>
                        %91 = scf.for %arg17 = %c0 to %76 step %c1 iter_args(%arg18 = %73) -> (tensor<?x?xf32>) {
                          %104 = arith.muli %arg17, %22 : index
                          %105 = loom.subview %arg0[%arg11, %38, %53, %39, %104] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %105, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %106 = loom.bufferize_to_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %107 = loom.subview %arg1[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %107, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %108 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                          %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%108 : tensor<?xf16>) outs(%85 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %82 : memref<?xf16>
                          %110 = loom.subview %arg2[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %110, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %111 = loom.bufferize_to_tensor %81[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%111 : tensor<?xf16>) outs(%88 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %81 : memref<?xf16>
                          %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %47, %109, %112 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%79 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %117 = arith.truncf %cst_0 : f64 to f32
                            %118 = arith.mulf %in_3, %117 : f32
                            %119 = arith.mulf %in_2, %117 : f32
                            %120 = arith.subf %119, %118 : f32
                            %121 = math.powf %cst, %120 : f32
                            %122 = arith.extf %in : f16 to f32
                            %123 = arith.mulf %122, %121 : f32
                            %124 = arith.mulf %123, %in_4 : f32
                            %125 = arith.truncf %124 : f32 to f16
                            linalg.yield %125 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?xf32>
                          loom.semaphore_give %84 : memref<?xf32>
                          %114 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %114, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %115 = loom.bufferize_to_tensor %90[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %116 = linalg.matmul ins(%113, %115 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          scf.yield %116 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %45 : memref<?xf32>
                        %92 = loom.alloc [1] on @L1 : memref<f16>
                        %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
                        %94 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        loom.copy %94, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %95 = loom.bufferize_to_tensor %93[] : memref<f16> -> tensor<f16>
                        %96 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                        %98 = loom.init_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %99 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %99, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %100 = loom.bufferize_to_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%91, %100, %95 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%98 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %104 = arith.extf %in_3 : f16 to f32
                          %105 = arith.extf %in_2 : f16 to f32
                          %106 = arith.mulf %105, %104 : f32
                          %107 = arith.addf %in, %106 : f32
                          %108 = arith.truncf %107 : f32 to f16
                          linalg.yield %108 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %93 : memref<f16>
                        loom.semaphore_give %69 : memref<?x?xf32>
                        %102 = loom.subview %arg7[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %103 = loom.bufferize_to_memref %101 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %103, %102 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %97 : memref<?x?xf16>
                      } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %25 = arith.ceildivui %c16, %23 : index
      %26 = arith.ceildivui %c256, %20 : index
      %27 = arith.ceildivui %c64, %21 : index
      %28 = arith.ceildivui %c8, %24 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %29 = arith.ceildivui %25, %c2 : index
                scf.for %arg13 = %c0 to %29 step %c1 {
                  %30 = arith.ceildivui %26, %c4 : index
                  scf.for %arg14 = %c0 to %30 step %c1 {
                    %31 = arith.ceildivui %27, %c2 : index
                    scf.for %arg15 = %c0 to %31 step %c1 {
                      %32 = arith.ceildivui %28, %c2 : index
                      scf.for %arg16 = %c0 to %32 step %c1 {
                        %33 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                        %34 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg9, %arg14)
                        %35 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                        %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg16)
                        %37 = arith.muli %33, %23 : index
                        %38 = arith.muli %36, %24 : index
                        %39 = arith.muli %34, %20 : index
                        %40 = loom.alloc [%20] on @L1 : memref<?xf16>
                        %41 = loom.semaphore_take %40 : memref<?xf16> -> memref<?xf16>
                        %42 = loom.subview %arg1[%arg11, %37, %38, %39] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %43 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %44 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %45 = loom.semaphore_take %44 : memref<?xf32> -> memref<?xf32>
                        %46 = loom.init_tensor %45[%20] : memref<?xf32> -> tensor<?xf32>
                        %47 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%46 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %104 = arith.extf %in : f16 to f32
                          linalg.yield %104 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %48 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %49 = loom.semaphore_take %48 : memref<?xf32> -> memref<?xf32>
                        %50 = loom.init_tensor %49[%20] : memref<?xf32> -> tensor<?xf32>
                        %51 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%47 : tensor<?xf32>) outs(%50 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %104 = arith.truncf %cst_0 : f64 to f32
                          %105 = arith.mulf %in, %104 : f32
                          %106 = math.powf %cst, %105 : f32
                          linalg.yield %106 : f32
                        } -> tensor<?xf32>
                        %52 = arith.muli %38, %c256 : index
                        %53 = arith.divui %37, %c16 : index
                        %54 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %55 = loom.semaphore_take %54 : memref<?x16xf16> -> memref<?x16xf16>
                        %56 = loom.subview %arg4[%arg11, %52, %53, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %56, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %57 = loom.bufferize_to_tensor %55[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %58 = arith.muli %35, %21 : index
                        %59 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %60 = loom.semaphore_take %59 : memref<?x16xf16> -> memref<?x16xf16>
                        %61 = loom.subview %arg5[%arg11, %38, %37, %58, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %61, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %62 = loom.bufferize_to_tensor %60[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %63 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %64 = loom.semaphore_take %63 : memref<16x?xf16> -> memref<16x?xf16>
                        %65 = loom.init_tensor %64[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%62 : tensor<?x16xf16>) outs(%65 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %60 : memref<?x16xf16>
                        %66 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %67 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %68 = loom.init_tensor %67[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %69 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %70 = loom.init_tensor %69[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %71 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %72 = linalg.matmul ins(%57, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%71 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %64 : memref<16x?xf16>
                        loom.semaphore_give %55 : memref<?x16xf16>
                        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %51 : tensor<?x?xf32>, tensor<?xf32>) outs(%70 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %104 = arith.mulf %in, %in_2 : f32
                          linalg.yield %104 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %67 : memref<?x?xf32>
                        loom.semaphore_give %49 : memref<?xf32>
                        %74 = arith.addi %34, %c1 : index
                        %75 = arith.muli %74, %20 : index
                        %76 = arith.ceildivui %75, %22 : index
                        %77 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                        %79 = loom.init_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %80 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %81 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %82 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %83 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %84 = loom.semaphore_take %83 : memref<?xf32> -> memref<?xf32>
                        %85 = loom.init_tensor %84[%22] : memref<?xf32> -> tensor<?xf32>
                        %86 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %87 = loom.semaphore_take %86 : memref<?xf32> -> memref<?xf32>
                        %88 = loom.init_tensor %87[%22] : memref<?xf32> -> tensor<?xf32>
                        %89 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %90 = loom.semaphore_take %89 : memref<?x?xf16> -> memref<?x?xf16>
                        %91 = scf.for %arg17 = %c0 to %76 step %c1 iter_args(%arg18 = %73) -> (tensor<?x?xf32>) {
                          %104 = arith.muli %arg17, %22 : index
                          %105 = loom.subview %arg0[%arg11, %38, %53, %39, %104] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %105, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %106 = loom.bufferize_to_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %107 = loom.subview %arg1[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %107, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %108 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                          %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%108 : tensor<?xf16>) outs(%85 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %82 : memref<?xf16>
                          %110 = loom.subview %arg2[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %110, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %111 = loom.bufferize_to_tensor %81[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%111 : tensor<?xf16>) outs(%88 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %81 : memref<?xf16>
                          %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %47, %109, %112 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%79 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %117 = arith.truncf %cst_0 : f64 to f32
                            %118 = arith.mulf %in_3, %117 : f32
                            %119 = arith.mulf %in_2, %117 : f32
                            %120 = arith.subf %119, %118 : f32
                            %121 = math.powf %cst, %120 : f32
                            %122 = arith.extf %in : f16 to f32
                            %123 = arith.mulf %122, %121 : f32
                            %124 = arith.mulf %123, %in_4 : f32
                            %125 = arith.truncf %124 : f32 to f16
                            linalg.yield %125 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?xf32>
                          loom.semaphore_give %84 : memref<?xf32>
                          %114 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %114, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %115 = loom.bufferize_to_tensor %90[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %116 = linalg.matmul ins(%113, %115 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          scf.yield %116 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %45 : memref<?xf32>
                        %92 = loom.alloc [1] on @L1 : memref<f16>
                        %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
                        %94 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        loom.copy %94, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %95 = loom.bufferize_to_tensor %93[] : memref<f16> -> tensor<f16>
                        %96 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                        %98 = loom.init_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %99 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %99, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %100 = loom.bufferize_to_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%91, %100, %95 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%98 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %104 = arith.extf %in_3 : f16 to f32
                          %105 = arith.extf %in_2 : f16 to f32
                          %106 = arith.mulf %105, %104 : f32
                          %107 = arith.addf %in, %106 : f32
                          %108 = arith.truncf %107 : f32 to f16
                          linalg.yield %108 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %93 : memref<f16>
                        loom.semaphore_give %69 : memref<?x?xf32>
                        %102 = loom.subview %arg7[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %103 = loom.bufferize_to_memref %101 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %103, %102 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %97 : memref<?x?xf16>
                      } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %25 = arith.ceildivui %c16, %23 : index
      %26 = arith.ceildivui %c256, %20 : index
      %27 = arith.ceildivui %c64, %21 : index
      %28 = arith.ceildivui %c8, %24 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %29 = arith.ceildivui %25, %c2 : index
                scf.for %arg13 = %c0 to %29 step %c1 {
                  %30 = arith.ceildivui %26, %c4 : index
                  scf.for %arg14 = %c0 to %30 step %c1 {
                    %31 = arith.ceildivui %27, %c2 : index
                    scf.for %arg15 = %c0 to %31 step %c1 {
                      %32 = arith.ceildivui %28, %c2 : index
                      scf.for %arg16 = %c0 to %32 step %c1 {
                        %33 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                        %34 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg9, %arg14)
                        %35 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                        %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg16)
                        %37 = arith.muli %33, %23 : index
                        %38 = arith.muli %36, %24 : index
                        %39 = arith.muli %34, %20 : index
                        %40 = loom.alloc [%20] on @L1 : memref<?xf16>
                        %41 = loom.semaphore_take %40 : memref<?xf16> -> memref<?xf16>
                        %42 = loom.subview %arg1[%arg11, %37, %38, %39] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %43 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %44 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %45 = loom.semaphore_take %44 : memref<?xf32> -> memref<?xf32>
                        %46 = loom.init_tensor %45[%20] : memref<?xf32> -> tensor<?xf32>
                        %47 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%46 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %104 = arith.extf %in : f16 to f32
                          linalg.yield %104 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %48 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %49 = loom.semaphore_take %48 : memref<?xf32> -> memref<?xf32>
                        %50 = loom.init_tensor %49[%20] : memref<?xf32> -> tensor<?xf32>
                        %51 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%47 : tensor<?xf32>) outs(%50 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %104 = arith.truncf %cst_0 : f64 to f32
                          %105 = arith.mulf %in, %104 : f32
                          %106 = math.powf %cst, %105 : f32
                          linalg.yield %106 : f32
                        } -> tensor<?xf32>
                        %52 = arith.muli %38, %c256 : index
                        %53 = arith.divui %37, %c16 : index
                        %54 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %55 = loom.semaphore_take %54 : memref<?x16xf16> -> memref<?x16xf16>
                        %56 = loom.subview %arg4[%arg11, %52, %53, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %56, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %57 = loom.bufferize_to_tensor %55[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %58 = arith.muli %35, %21 : index
                        %59 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %60 = loom.semaphore_take %59 : memref<?x16xf16> -> memref<?x16xf16>
                        %61 = loom.subview %arg5[%arg11, %38, %37, %58, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %61, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %62 = loom.bufferize_to_tensor %60[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %63 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %64 = loom.semaphore_take %63 : memref<16x?xf16> -> memref<16x?xf16>
                        %65 = loom.init_tensor %64[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%62 : tensor<?x16xf16>) outs(%65 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %60 : memref<?x16xf16>
                        %66 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %67 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %68 = loom.init_tensor %67[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %69 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %70 = loom.init_tensor %69[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %71 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %72 = linalg.matmul ins(%57, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%71 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %64 : memref<16x?xf16>
                        loom.semaphore_give %55 : memref<?x16xf16>
                        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %51 : tensor<?x?xf32>, tensor<?xf32>) outs(%70 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %104 = arith.mulf %in, %in_2 : f32
                          linalg.yield %104 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %67 : memref<?x?xf32>
                        loom.semaphore_give %49 : memref<?xf32>
                        %74 = arith.addi %34, %c1 : index
                        %75 = arith.muli %74, %20 : index
                        %76 = arith.ceildivui %75, %22 : index
                        %77 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                        %79 = loom.init_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %80 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %81 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %82 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %83 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %84 = loom.semaphore_take %83 : memref<?xf32> -> memref<?xf32>
                        %85 = loom.init_tensor %84[%22] : memref<?xf32> -> tensor<?xf32>
                        %86 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %87 = loom.semaphore_take %86 : memref<?xf32> -> memref<?xf32>
                        %88 = loom.init_tensor %87[%22] : memref<?xf32> -> tensor<?xf32>
                        %89 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %90 = loom.semaphore_take %89 : memref<?x?xf16> -> memref<?x?xf16>
                        %91 = scf.for %arg17 = %c0 to %76 step %c1 iter_args(%arg18 = %73) -> (tensor<?x?xf32>) {
                          %104 = arith.muli %arg17, %22 : index
                          %105 = loom.subview %arg0[%arg11, %38, %53, %39, %104] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %105, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %106 = loom.bufferize_to_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %107 = loom.subview %arg1[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %107, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %108 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                          %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%108 : tensor<?xf16>) outs(%85 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %82 : memref<?xf16>
                          %110 = loom.subview %arg2[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %110, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %111 = loom.bufferize_to_tensor %81[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%111 : tensor<?xf16>) outs(%88 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %81 : memref<?xf16>
                          %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %47, %109, %112 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%79 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %117 = arith.truncf %cst_0 : f64 to f32
                            %118 = arith.mulf %in_3, %117 : f32
                            %119 = arith.mulf %in_2, %117 : f32
                            %120 = arith.subf %119, %118 : f32
                            %121 = math.powf %cst, %120 : f32
                            %122 = arith.extf %in : f16 to f32
                            %123 = arith.mulf %122, %121 : f32
                            %124 = arith.mulf %123, %in_4 : f32
                            %125 = arith.truncf %124 : f32 to f16
                            linalg.yield %125 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?xf32>
                          loom.semaphore_give %84 : memref<?xf32>
                          %114 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %114, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %115 = loom.bufferize_to_tensor %90[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %116 = linalg.matmul ins(%113, %115 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          scf.yield %116 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %45 : memref<?xf32>
                        %92 = loom.alloc [1] on @L1 : memref<f16>
                        %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
                        %94 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        loom.copy %94, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %95 = loom.bufferize_to_tensor %93[] : memref<f16> -> tensor<f16>
                        %96 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                        %98 = loom.init_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %99 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %99, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %100 = loom.bufferize_to_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%91, %100, %95 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%98 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %104 = arith.extf %in_3 : f16 to f32
                          %105 = arith.extf %in_2 : f16 to f32
                          %106 = arith.mulf %105, %104 : f32
                          %107 = arith.addf %in, %106 : f32
                          %108 = arith.truncf %107 : f32 to f16
                          linalg.yield %108 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %93 : memref<f16>
                        loom.semaphore_give %69 : memref<?x?xf32>
                        %102 = loom.subview %arg7[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %103 = loom.bufferize_to_memref %101 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %103, %102 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %97 : memref<?x?xf16>
                      } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %25 = arith.ceildivui %c16, %23 : index
      %26 = arith.ceildivui %c256, %20 : index
      %27 = arith.ceildivui %c64, %21 : index
      %28 = arith.ceildivui %c8, %24 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                %29 = arith.ceildivui %25, %c2 : index
                scf.for %arg13 = %c0 to %29 step %c1 {
                  %30 = arith.ceildivui %26, %c2 : index
                  scf.for %arg14 = %c0 to %30 step %c1 {
                    %31 = arith.ceildivui %27, %c2 : index
                    scf.for %arg15 = %c0 to %31 step %c1 {
                      %32 = arith.ceildivui %28, %c4 : index
                      scf.for %arg16 = %c0 to %32 step %c1 {
                        %33 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                        %34 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                        %35 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                        %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg12, %arg16)
                        %37 = arith.muli %33, %23 : index
                        %38 = arith.muli %36, %24 : index
                        %39 = arith.muli %34, %20 : index
                        %40 = loom.alloc [%20] on @L1 : memref<?xf16>
                        %41 = loom.semaphore_take %40 : memref<?xf16> -> memref<?xf16>
                        %42 = loom.subview %arg1[%arg11, %37, %38, %39] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %43 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %44 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %45 = loom.semaphore_take %44 : memref<?xf32> -> memref<?xf32>
                        %46 = loom.init_tensor %45[%20] : memref<?xf32> -> tensor<?xf32>
                        %47 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%46 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %104 = arith.extf %in : f16 to f32
                          linalg.yield %104 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %48 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %49 = loom.semaphore_take %48 : memref<?xf32> -> memref<?xf32>
                        %50 = loom.init_tensor %49[%20] : memref<?xf32> -> tensor<?xf32>
                        %51 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%47 : tensor<?xf32>) outs(%50 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %104 = arith.truncf %cst_0 : f64 to f32
                          %105 = arith.mulf %in, %104 : f32
                          %106 = math.powf %cst, %105 : f32
                          linalg.yield %106 : f32
                        } -> tensor<?xf32>
                        %52 = arith.muli %38, %c256 : index
                        %53 = arith.divui %37, %c16 : index
                        %54 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %55 = loom.semaphore_take %54 : memref<?x16xf16> -> memref<?x16xf16>
                        %56 = loom.subview %arg4[%arg11, %52, %53, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %56, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %57 = loom.bufferize_to_tensor %55[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %58 = arith.muli %35, %21 : index
                        %59 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %60 = loom.semaphore_take %59 : memref<?x16xf16> -> memref<?x16xf16>
                        %61 = loom.subview %arg5[%arg11, %38, %37, %58, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %61, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %62 = loom.bufferize_to_tensor %60[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %63 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %64 = loom.semaphore_take %63 : memref<16x?xf16> -> memref<16x?xf16>
                        %65 = loom.init_tensor %64[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%62 : tensor<?x16xf16>) outs(%65 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %60 : memref<?x16xf16>
                        %66 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %67 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %68 = loom.init_tensor %67[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %69 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %70 = loom.init_tensor %69[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %71 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %72 = linalg.matmul ins(%57, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%71 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %64 : memref<16x?xf16>
                        loom.semaphore_give %55 : memref<?x16xf16>
                        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %51 : tensor<?x?xf32>, tensor<?xf32>) outs(%70 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %104 = arith.mulf %in, %in_2 : f32
                          linalg.yield %104 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %67 : memref<?x?xf32>
                        loom.semaphore_give %49 : memref<?xf32>
                        %74 = arith.addi %34, %c1 : index
                        %75 = arith.muli %74, %20 : index
                        %76 = arith.ceildivui %75, %22 : index
                        %77 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                        %79 = loom.init_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %80 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %81 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %82 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %83 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %84 = loom.semaphore_take %83 : memref<?xf32> -> memref<?xf32>
                        %85 = loom.init_tensor %84[%22] : memref<?xf32> -> tensor<?xf32>
                        %86 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %87 = loom.semaphore_take %86 : memref<?xf32> -> memref<?xf32>
                        %88 = loom.init_tensor %87[%22] : memref<?xf32> -> tensor<?xf32>
                        %89 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %90 = loom.semaphore_take %89 : memref<?x?xf16> -> memref<?x?xf16>
                        %91 = scf.for %arg17 = %c0 to %76 step %c1 iter_args(%arg18 = %73) -> (tensor<?x?xf32>) {
                          %104 = arith.muli %arg17, %22 : index
                          %105 = loom.subview %arg0[%arg11, %38, %53, %39, %104] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %105, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %106 = loom.bufferize_to_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %107 = loom.subview %arg1[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %107, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %108 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                          %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%108 : tensor<?xf16>) outs(%85 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %82 : memref<?xf16>
                          %110 = loom.subview %arg2[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %110, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %111 = loom.bufferize_to_tensor %81[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%111 : tensor<?xf16>) outs(%88 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %81 : memref<?xf16>
                          %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %47, %109, %112 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%79 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %117 = arith.truncf %cst_0 : f64 to f32
                            %118 = arith.mulf %in_3, %117 : f32
                            %119 = arith.mulf %in_2, %117 : f32
                            %120 = arith.subf %119, %118 : f32
                            %121 = math.powf %cst, %120 : f32
                            %122 = arith.extf %in : f16 to f32
                            %123 = arith.mulf %122, %121 : f32
                            %124 = arith.mulf %123, %in_4 : f32
                            %125 = arith.truncf %124 : f32 to f16
                            linalg.yield %125 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?xf32>
                          loom.semaphore_give %84 : memref<?xf32>
                          %114 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %114, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %115 = loom.bufferize_to_tensor %90[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %116 = linalg.matmul ins(%113, %115 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          scf.yield %116 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %45 : memref<?xf32>
                        %92 = loom.alloc [1] on @L1 : memref<f16>
                        %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
                        %94 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        loom.copy %94, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %95 = loom.bufferize_to_tensor %93[] : memref<f16> -> tensor<f16>
                        %96 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                        %98 = loom.init_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %99 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %99, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %100 = loom.bufferize_to_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%91, %100, %95 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%98 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %104 = arith.extf %in_3 : f16 to f32
                          %105 = arith.extf %in_2 : f16 to f32
                          %106 = arith.mulf %105, %104 : f32
                          %107 = arith.addf %in, %106 : f32
                          %108 = arith.truncf %107 : f32 to f16
                          linalg.yield %108 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %93 : memref<f16>
                        loom.semaphore_give %69 : memref<?x?xf32>
                        %102 = loom.subview %arg7[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %103 = loom.bufferize_to_memref %101 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %103, %102 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %97 : memref<?x?xf16>
                      } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %25 = arith.ceildivui %c16, %23 : index
      %26 = arith.ceildivui %c256, %20 : index
      %27 = arith.ceildivui %c64, %21 : index
      %28 = arith.ceildivui %c8, %24 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                %29 = arith.ceildivui %25, %c2 : index
                scf.for %arg13 = %c0 to %29 step %c1 {
                  %30 = arith.ceildivui %26, %c2 : index
                  scf.for %arg14 = %c0 to %30 step %c1 {
                    %31 = arith.ceildivui %27, %c2 : index
                    scf.for %arg15 = %c0 to %31 step %c1 {
                      %32 = arith.ceildivui %28, %c2 : index
                      scf.for %arg16 = %c0 to %32 step %c1 {
                        %33 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                        %34 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                        %35 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg10, %arg15)
                        %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg16)
                        %37 = arith.muli %33, %23 : index
                        %38 = arith.muli %36, %24 : index
                        %39 = arith.muli %34, %20 : index
                        %40 = loom.alloc [%20] on @L1 : memref<?xf16>
                        %41 = loom.semaphore_take %40 : memref<?xf16> -> memref<?xf16>
                        %42 = loom.subview %arg1[%arg11, %37, %38, %39] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %43 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %44 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %45 = loom.semaphore_take %44 : memref<?xf32> -> memref<?xf32>
                        %46 = loom.init_tensor %45[%20] : memref<?xf32> -> tensor<?xf32>
                        %47 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%46 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %104 = arith.extf %in : f16 to f32
                          linalg.yield %104 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %48 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %49 = loom.semaphore_take %48 : memref<?xf32> -> memref<?xf32>
                        %50 = loom.init_tensor %49[%20] : memref<?xf32> -> tensor<?xf32>
                        %51 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%47 : tensor<?xf32>) outs(%50 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %104 = arith.truncf %cst_0 : f64 to f32
                          %105 = arith.mulf %in, %104 : f32
                          %106 = math.powf %cst, %105 : f32
                          linalg.yield %106 : f32
                        } -> tensor<?xf32>
                        %52 = arith.muli %38, %c256 : index
                        %53 = arith.divui %37, %c16 : index
                        %54 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %55 = loom.semaphore_take %54 : memref<?x16xf16> -> memref<?x16xf16>
                        %56 = loom.subview %arg4[%arg11, %52, %53, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %56, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %57 = loom.bufferize_to_tensor %55[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %58 = arith.muli %35, %21 : index
                        %59 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %60 = loom.semaphore_take %59 : memref<?x16xf16> -> memref<?x16xf16>
                        %61 = loom.subview %arg5[%arg11, %38, %37, %58, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %61, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %62 = loom.bufferize_to_tensor %60[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %63 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %64 = loom.semaphore_take %63 : memref<16x?xf16> -> memref<16x?xf16>
                        %65 = loom.init_tensor %64[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%62 : tensor<?x16xf16>) outs(%65 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %60 : memref<?x16xf16>
                        %66 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %67 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %68 = loom.init_tensor %67[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %69 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %70 = loom.init_tensor %69[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %71 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %72 = linalg.matmul ins(%57, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%71 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %64 : memref<16x?xf16>
                        loom.semaphore_give %55 : memref<?x16xf16>
                        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %51 : tensor<?x?xf32>, tensor<?xf32>) outs(%70 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %104 = arith.mulf %in, %in_2 : f32
                          linalg.yield %104 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %67 : memref<?x?xf32>
                        loom.semaphore_give %49 : memref<?xf32>
                        %74 = arith.addi %34, %c1 : index
                        %75 = arith.muli %74, %20 : index
                        %76 = arith.ceildivui %75, %22 : index
                        %77 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                        %79 = loom.init_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %80 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %81 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %82 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %83 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %84 = loom.semaphore_take %83 : memref<?xf32> -> memref<?xf32>
                        %85 = loom.init_tensor %84[%22] : memref<?xf32> -> tensor<?xf32>
                        %86 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %87 = loom.semaphore_take %86 : memref<?xf32> -> memref<?xf32>
                        %88 = loom.init_tensor %87[%22] : memref<?xf32> -> tensor<?xf32>
                        %89 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %90 = loom.semaphore_take %89 : memref<?x?xf16> -> memref<?x?xf16>
                        %91 = scf.for %arg17 = %c0 to %76 step %c1 iter_args(%arg18 = %73) -> (tensor<?x?xf32>) {
                          %104 = arith.muli %arg17, %22 : index
                          %105 = loom.subview %arg0[%arg11, %38, %53, %39, %104] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %105, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %106 = loom.bufferize_to_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %107 = loom.subview %arg1[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %107, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %108 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                          %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%108 : tensor<?xf16>) outs(%85 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %82 : memref<?xf16>
                          %110 = loom.subview %arg2[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %110, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %111 = loom.bufferize_to_tensor %81[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%111 : tensor<?xf16>) outs(%88 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %81 : memref<?xf16>
                          %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %47, %109, %112 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%79 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %117 = arith.truncf %cst_0 : f64 to f32
                            %118 = arith.mulf %in_3, %117 : f32
                            %119 = arith.mulf %in_2, %117 : f32
                            %120 = arith.subf %119, %118 : f32
                            %121 = math.powf %cst, %120 : f32
                            %122 = arith.extf %in : f16 to f32
                            %123 = arith.mulf %122, %121 : f32
                            %124 = arith.mulf %123, %in_4 : f32
                            %125 = arith.truncf %124 : f32 to f16
                            linalg.yield %125 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?xf32>
                          loom.semaphore_give %84 : memref<?xf32>
                          %114 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %114, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %115 = loom.bufferize_to_tensor %90[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %116 = linalg.matmul ins(%113, %115 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          scf.yield %116 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %45 : memref<?xf32>
                        %92 = loom.alloc [1] on @L1 : memref<f16>
                        %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
                        %94 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        loom.copy %94, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %95 = loom.bufferize_to_tensor %93[] : memref<f16> -> tensor<f16>
                        %96 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                        %98 = loom.init_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %99 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %99, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %100 = loom.bufferize_to_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%91, %100, %95 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%98 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %104 = arith.extf %in_3 : f16 to f32
                          %105 = arith.extf %in_2 : f16 to f32
                          %106 = arith.mulf %105, %104 : f32
                          %107 = arith.addf %in, %106 : f32
                          %108 = arith.truncf %107 : f32 to f16
                          linalg.yield %108 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %93 : memref<f16>
                        loom.semaphore_give %69 : memref<?x?xf32>
                        %102 = loom.subview %arg7[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %103 = loom.bufferize_to_memref %101 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %103, %102 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %97 : memref<?x?xf16>
                      } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %25 = arith.ceildivui %c16, %23 : index
      %26 = arith.ceildivui %c256, %20 : index
      %27 = arith.ceildivui %c64, %21 : index
      %28 = arith.ceildivui %c8, %24 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %29 = arith.ceildivui %25, %c2 : index
                scf.for %arg13 = %c0 to %29 step %c1 {
                  %30 = arith.ceildivui %26, %c2 : index
                  scf.for %arg14 = %c0 to %30 step %c1 {
                    %31 = arith.ceildivui %27, %c4 : index
                    scf.for %arg15 = %c0 to %31 step %c1 {
                      %32 = arith.ceildivui %28, %c2 : index
                      scf.for %arg16 = %c0 to %32 step %c1 {
                        %33 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                        %34 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                        %35 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg10, %arg15)
                        %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg16)
                        %37 = arith.muli %33, %23 : index
                        %38 = arith.muli %36, %24 : index
                        %39 = arith.muli %34, %20 : index
                        %40 = loom.alloc [%20] on @L1 : memref<?xf16>
                        %41 = loom.semaphore_take %40 : memref<?xf16> -> memref<?xf16>
                        %42 = loom.subview %arg1[%arg11, %37, %38, %39] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %43 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %44 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %45 = loom.semaphore_take %44 : memref<?xf32> -> memref<?xf32>
                        %46 = loom.init_tensor %45[%20] : memref<?xf32> -> tensor<?xf32>
                        %47 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%46 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %104 = arith.extf %in : f16 to f32
                          linalg.yield %104 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %48 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %49 = loom.semaphore_take %48 : memref<?xf32> -> memref<?xf32>
                        %50 = loom.init_tensor %49[%20] : memref<?xf32> -> tensor<?xf32>
                        %51 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%47 : tensor<?xf32>) outs(%50 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %104 = arith.truncf %cst_0 : f64 to f32
                          %105 = arith.mulf %in, %104 : f32
                          %106 = math.powf %cst, %105 : f32
                          linalg.yield %106 : f32
                        } -> tensor<?xf32>
                        %52 = arith.muli %38, %c256 : index
                        %53 = arith.divui %37, %c16 : index
                        %54 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %55 = loom.semaphore_take %54 : memref<?x16xf16> -> memref<?x16xf16>
                        %56 = loom.subview %arg4[%arg11, %52, %53, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %56, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %57 = loom.bufferize_to_tensor %55[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %58 = arith.muli %35, %21 : index
                        %59 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %60 = loom.semaphore_take %59 : memref<?x16xf16> -> memref<?x16xf16>
                        %61 = loom.subview %arg5[%arg11, %38, %37, %58, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %61, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %62 = loom.bufferize_to_tensor %60[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %63 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %64 = loom.semaphore_take %63 : memref<16x?xf16> -> memref<16x?xf16>
                        %65 = loom.init_tensor %64[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%62 : tensor<?x16xf16>) outs(%65 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %60 : memref<?x16xf16>
                        %66 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %67 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %68 = loom.init_tensor %67[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %69 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %70 = loom.init_tensor %69[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %71 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %72 = linalg.matmul ins(%57, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%71 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %64 : memref<16x?xf16>
                        loom.semaphore_give %55 : memref<?x16xf16>
                        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %51 : tensor<?x?xf32>, tensor<?xf32>) outs(%70 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %104 = arith.mulf %in, %in_2 : f32
                          linalg.yield %104 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %67 : memref<?x?xf32>
                        loom.semaphore_give %49 : memref<?xf32>
                        %74 = arith.addi %34, %c1 : index
                        %75 = arith.muli %74, %20 : index
                        %76 = arith.ceildivui %75, %22 : index
                        %77 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                        %79 = loom.init_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %80 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %81 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %82 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %83 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %84 = loom.semaphore_take %83 : memref<?xf32> -> memref<?xf32>
                        %85 = loom.init_tensor %84[%22] : memref<?xf32> -> tensor<?xf32>
                        %86 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %87 = loom.semaphore_take %86 : memref<?xf32> -> memref<?xf32>
                        %88 = loom.init_tensor %87[%22] : memref<?xf32> -> tensor<?xf32>
                        %89 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %90 = loom.semaphore_take %89 : memref<?x?xf16> -> memref<?x?xf16>
                        %91 = scf.for %arg17 = %c0 to %76 step %c1 iter_args(%arg18 = %73) -> (tensor<?x?xf32>) {
                          %104 = arith.muli %arg17, %22 : index
                          %105 = loom.subview %arg0[%arg11, %38, %53, %39, %104] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %105, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %106 = loom.bufferize_to_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %107 = loom.subview %arg1[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %107, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %108 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                          %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%108 : tensor<?xf16>) outs(%85 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %82 : memref<?xf16>
                          %110 = loom.subview %arg2[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %110, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %111 = loom.bufferize_to_tensor %81[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%111 : tensor<?xf16>) outs(%88 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %81 : memref<?xf16>
                          %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %47, %109, %112 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%79 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %117 = arith.truncf %cst_0 : f64 to f32
                            %118 = arith.mulf %in_3, %117 : f32
                            %119 = arith.mulf %in_2, %117 : f32
                            %120 = arith.subf %119, %118 : f32
                            %121 = math.powf %cst, %120 : f32
                            %122 = arith.extf %in : f16 to f32
                            %123 = arith.mulf %122, %121 : f32
                            %124 = arith.mulf %123, %in_4 : f32
                            %125 = arith.truncf %124 : f32 to f16
                            linalg.yield %125 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?xf32>
                          loom.semaphore_give %84 : memref<?xf32>
                          %114 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %114, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %115 = loom.bufferize_to_tensor %90[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %116 = linalg.matmul ins(%113, %115 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          scf.yield %116 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %45 : memref<?xf32>
                        %92 = loom.alloc [1] on @L1 : memref<f16>
                        %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
                        %94 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        loom.copy %94, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %95 = loom.bufferize_to_tensor %93[] : memref<f16> -> tensor<f16>
                        %96 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                        %98 = loom.init_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %99 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %99, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %100 = loom.bufferize_to_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%91, %100, %95 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%98 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %104 = arith.extf %in_3 : f16 to f32
                          %105 = arith.extf %in_2 : f16 to f32
                          %106 = arith.mulf %105, %104 : f32
                          %107 = arith.addf %in, %106 : f32
                          %108 = arith.truncf %107 : f32 to f16
                          linalg.yield %108 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %93 : memref<f16>
                        loom.semaphore_give %69 : memref<?x?xf32>
                        %102 = loom.subview %arg7[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %103 = loom.bufferize_to_memref %101 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %103, %102 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %97 : memref<?x?xf16>
                      } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %20 = loom.sym @tile_m {upper_bound = 256 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_h {upper_bound = 16 : index} : index
      %24 = loom.sym @tile_c {upper_bound = 8 : index} : index
      %25 = arith.ceildivui %c16, %23 : index
      %26 = arith.ceildivui %c256, %20 : index
      %27 = arith.ceildivui %c64, %21 : index
      %28 = arith.ceildivui %c8, %24 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %29 = arith.ceildivui %25, %c2 : index
                scf.for %arg13 = %c0 to %29 step %c1 {
                  %30 = arith.ceildivui %26, %c2 : index
                  scf.for %arg14 = %c0 to %30 step %c1 {
                    %31 = arith.ceildivui %27, %c4 : index
                    scf.for %arg15 = %c0 to %31 step %c1 {
                      %32 = arith.ceildivui %28, %c2 : index
                      scf.for %arg16 = %c0 to %32 step %c1 {
                        %33 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg8, %arg13)
                        %34 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg14)
                        %35 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg10, %arg15)
                        %36 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg12, %arg16)
                        %37 = arith.muli %33, %23 : index
                        %38 = arith.muli %36, %24 : index
                        %39 = arith.muli %34, %20 : index
                        %40 = loom.alloc [%20] on @L1 : memref<?xf16>
                        %41 = loom.semaphore_take %40 : memref<?xf16> -> memref<?xf16>
                        %42 = loom.subview %arg1[%arg11, %37, %38, %39] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %43 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %44 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %45 = loom.semaphore_take %44 : memref<?xf32> -> memref<?xf32>
                        %46 = loom.init_tensor %45[%20] : memref<?xf32> -> tensor<?xf32>
                        %47 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%46 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %104 = arith.extf %in : f16 to f32
                          linalg.yield %104 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %48 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %49 = loom.semaphore_take %48 : memref<?xf32> -> memref<?xf32>
                        %50 = loom.init_tensor %49[%20] : memref<?xf32> -> tensor<?xf32>
                        %51 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%47 : tensor<?xf32>) outs(%50 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %104 = arith.truncf %cst_0 : f64 to f32
                          %105 = arith.mulf %in, %104 : f32
                          %106 = math.powf %cst, %105 : f32
                          linalg.yield %106 : f32
                        } -> tensor<?xf32>
                        %52 = arith.muli %38, %c256 : index
                        %53 = arith.divui %37, %c16 : index
                        %54 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %55 = loom.semaphore_take %54 : memref<?x16xf16> -> memref<?x16xf16>
                        %56 = loom.subview %arg4[%arg11, %52, %53, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %56, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %57 = loom.bufferize_to_tensor %55[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %58 = arith.muli %35, %21 : index
                        %59 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %60 = loom.semaphore_take %59 : memref<?x16xf16> -> memref<?x16xf16>
                        %61 = loom.subview %arg5[%arg11, %38, %37, %58, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        loom.copy %61, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %62 = loom.bufferize_to_tensor %60[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %63 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %64 = loom.semaphore_take %63 : memref<16x?xf16> -> memref<16x?xf16>
                        %65 = loom.init_tensor %64[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%62 : tensor<?x16xf16>) outs(%65 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %60 : memref<?x16xf16>
                        %66 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %67 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %68 = loom.init_tensor %67[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %69 = loom.semaphore_take %66 : memref<?x?xf32> -> memref<?x?xf32>
                        %70 = loom.init_tensor %69[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %71 = linalg.fill ins(%cst_1 : f32) outs(%68 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %72 = linalg.matmul ins(%57, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%71 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %64 : memref<16x?xf16>
                        loom.semaphore_give %55 : memref<?x16xf16>
                        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%72, %51 : tensor<?x?xf32>, tensor<?xf32>) outs(%70 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %104 = arith.mulf %in, %in_2 : f32
                          linalg.yield %104 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %67 : memref<?x?xf32>
                        loom.semaphore_give %49 : memref<?xf32>
                        %74 = arith.addi %34, %c1 : index
                        %75 = arith.muli %74, %20 : index
                        %76 = arith.ceildivui %75, %22 : index
                        %77 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
                        %79 = loom.init_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %80 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %81 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %82 = loom.semaphore_take %80 : memref<?xf16> -> memref<?xf16>
                        %83 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %84 = loom.semaphore_take %83 : memref<?xf32> -> memref<?xf32>
                        %85 = loom.init_tensor %84[%22] : memref<?xf32> -> tensor<?xf32>
                        %86 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %87 = loom.semaphore_take %86 : memref<?xf32> -> memref<?xf32>
                        %88 = loom.init_tensor %87[%22] : memref<?xf32> -> tensor<?xf32>
                        %89 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %90 = loom.semaphore_take %89 : memref<?x?xf16> -> memref<?x?xf16>
                        %91 = scf.for %arg17 = %c0 to %76 step %c1 iter_args(%arg18 = %73) -> (tensor<?x?xf32>) {
                          %104 = arith.muli %arg17, %22 : index
                          %105 = loom.subview %arg0[%arg11, %38, %53, %39, %104] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %105, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %106 = loom.bufferize_to_tensor %78[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %107 = loom.subview %arg1[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %107, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %108 = loom.bufferize_to_tensor %82[%22] : memref<?xf16> -> tensor<?xf16>
                          %109 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%108 : tensor<?xf16>) outs(%85 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %82 : memref<?xf16>
                          %110 = loom.subview %arg2[%arg11, %37, %38, %104] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %110, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %111 = loom.bufferize_to_tensor %81[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%111 : tensor<?xf16>) outs(%88 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %117 = arith.extf %in : f16 to f32
                            linalg.yield %117 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %81 : memref<?xf16>
                          %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %47, %109, %112 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%79 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %117 = arith.truncf %cst_0 : f64 to f32
                            %118 = arith.mulf %in_3, %117 : f32
                            %119 = arith.mulf %in_2, %117 : f32
                            %120 = arith.subf %119, %118 : f32
                            %121 = math.powf %cst, %120 : f32
                            %122 = arith.extf %in : f16 to f32
                            %123 = arith.mulf %122, %121 : f32
                            %124 = arith.mulf %123, %in_4 : f32
                            %125 = arith.truncf %124 : f32 to f16
                            linalg.yield %125 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?xf32>
                          loom.semaphore_give %84 : memref<?xf32>
                          %114 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %114, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %115 = loom.bufferize_to_tensor %90[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %116 = linalg.matmul ins(%113, %115 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          loom.semaphore_give %78 : memref<?x?xf16>
                          scf.yield %116 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %45 : memref<?xf32>
                        %92 = loom.alloc [1] on @L1 : memref<f16>
                        %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
                        %94 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        loom.copy %94, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %95 = loom.bufferize_to_tensor %93[] : memref<f16> -> tensor<f16>
                        %96 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %97 = loom.semaphore_take %96 : memref<?x?xf16> -> memref<?x?xf16>
                        %98 = loom.init_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %99 = loom.subview %arg3[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %99, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %100 = loom.bufferize_to_tensor %97[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%91, %100, %95 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%98 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %104 = arith.extf %in_3 : f16 to f32
                          %105 = arith.extf %in_2 : f16 to f32
                          %106 = arith.mulf %105, %104 : f32
                          %107 = arith.addf %in, %106 : f32
                          %108 = arith.truncf %107 : f32 to f16
                          linalg.yield %108 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %93 : memref<f16>
                        loom.semaphore_give %69 : memref<?x?xf32>
                        %102 = loom.subview %arg7[%arg11, %52, %37, %58] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %103 = loom.bufferize_to_memref %101 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %103, %102 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %97 : memref<?x?xf16>
                      } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<temporal>}
                    } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
                  } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
                } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
