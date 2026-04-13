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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
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
                        %43 = arith.muli %arg11, %c2 : index
                        %44 = arith.addi %arg9, %43 : index
                        %45 = arith.muli %arg12, %c2 : index
                        %46 = arith.muli %arg8, %c4 : index
                        %47 = arith.addi %45, %46 : index
                        %48 = arith.addi %45, %c1 : index
                        %49 = arith.addi %48, %46 : index
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%44, %47], LR : [%44, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %50 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %51 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %52 = loom.semaphore_take %51 : memref<?xf32> -> memref<?xf32>
                        %53 = loom.init_tensor %52[%20] : memref<?xf32> -> tensor<?xf32>
                        %54 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50 : tensor<?xf16>) outs(%53 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %115 = arith.extf %in : f16 to f32
                          linalg.yield %115 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %55 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %56 = loom.semaphore_take %55 : memref<?xf32> -> memref<?xf32>
                        %57 = loom.init_tensor %56[%20] : memref<?xf32> -> tensor<?xf32>
                        %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf32>) outs(%57 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %115 = arith.truncf %cst_0 : f64 to f32
                          %116 = arith.mulf %in, %115 : f32
                          %117 = math.powf %cst, %116 : f32
                          linalg.yield %117 : f32
                        } -> tensor<?xf32>
                        %59 = arith.muli %38, %c256 : index
                        %60 = arith.divui %37, %c16 : index
                        %61 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %62 = loom.semaphore_take %61 : memref<?x16xf16> -> memref<?x16xf16>
                        %63 = loom.subview %arg4[%arg11, %59, %60, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %64 = arith.addi %43, %c1 : index
                        loom.copy %63, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %65 = loom.bufferize_to_tensor %62[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %66 = arith.muli %35, %21 : index
                        %67 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %68 = loom.semaphore_take %67 : memref<?x16xf16> -> memref<?x16xf16>
                        %69 = loom.subview %arg5[%arg11, %38, %37, %66, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %70 = arith.addi %arg10, %45 : index
                        %71 = arith.addi %70, %46 : index
                        loom.copy %69, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %72 = loom.bufferize_to_tensor %68[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %73 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %74 = loom.semaphore_take %73 : memref<16x?xf16> -> memref<16x?xf16>
                        %75 = loom.init_tensor %74[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%72 : tensor<?x16xf16>) outs(%75 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %68 : memref<?x16xf16>
                        %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %77 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %79 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %81 = linalg.fill ins(%cst_1 : f32) outs(%78 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %82 = linalg.matmul ins(%65, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%81 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %74 : memref<16x?xf16>
                        loom.semaphore_give %62 : memref<?x16xf16>
                        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %58 : tensor<?x?xf32>, tensor<?xf32>) outs(%80 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %115 = arith.mulf %in, %in_2 : f32
                          linalg.yield %115 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %77 : memref<?x?xf32>
                        loom.semaphore_give %56 : memref<?xf32>
                        %84 = arith.addi %34, %c1 : index
                        %85 = arith.muli %84, %20 : index
                        %86 = arith.ceildivui %85, %22 : index
                        %87 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                        %89 = loom.init_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %90 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %91 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %92 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %93 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %94 = loom.semaphore_take %93 : memref<?xf32> -> memref<?xf32>
                        %95 = loom.init_tensor %94[%22] : memref<?xf32> -> tensor<?xf32>
                        %96 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %97 = loom.semaphore_take %96 : memref<?xf32> -> memref<?xf32>
                        %98 = loom.init_tensor %97[%22] : memref<?xf32> -> tensor<?xf32>
                        %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                        %101 = scf.for %arg17 = %c0 to %86 step %c1 iter_args(%arg18 = %83) -> (tensor<?x?xf32>) {
                          %115 = arith.muli %arg17, %22 : index
                          %116 = loom.subview %arg0[%arg11, %38, %60, %39, %115] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %116, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%44, %47], LR : [%44, %49]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %117 = loom.bufferize_to_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %118 = loom.subview %arg1[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %118, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %119 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                          %120 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%119 : tensor<?xf16>) outs(%95 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %92 : memref<?xf16>
                          %121 = loom.subview %arg2[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %121, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %122 = loom.bufferize_to_tensor %91[%22] : memref<?xf16> -> tensor<?xf16>
                          %123 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%122 : tensor<?xf16>) outs(%98 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %91 : memref<?xf16>
                          %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %54, %120, %123 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%89 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %128 = arith.truncf %cst_0 : f64 to f32
                            %129 = arith.mulf %in_3, %128 : f32
                            %130 = arith.mulf %in_2, %128 : f32
                            %131 = arith.subf %130, %129 : f32
                            %132 = math.powf %cst, %131 : f32
                            %133 = arith.extf %in : f16 to f32
                            %134 = arith.mulf %133, %132 : f32
                            %135 = arith.mulf %134, %in_4 : f32
                            %136 = arith.truncf %135 : f32 to f16
                            linalg.yield %136 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %97 : memref<?xf32>
                          loom.semaphore_give %94 : memref<?xf32>
                          %125 = loom.subview %arg3[%arg11, %59, %37, %66] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %125, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %126 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %127 = linalg.matmul ins(%124, %126 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          loom.semaphore_give %88 : memref<?x?xf16>
                          scf.yield %127 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %52 : memref<?xf32>
                        %102 = loom.alloc [1] on @L1 : memref<f16>
                        %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                        %104 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        %105 = arith.addi %46, %c3 : index
                        loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %46], LR : [%c7, %105]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                        %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                        %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %110 = loom.subview %arg3[%arg11, %59, %37, %66] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %115 = arith.extf %in_3 : f16 to f32
                          %116 = arith.extf %in_2 : f16 to f32
                          %117 = arith.mulf %116, %115 : f32
                          %118 = arith.addf %in, %117 : f32
                          %119 = arith.truncf %118 : f32 to f16
                          linalg.yield %119 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %103 : memref<f16>
                        loom.semaphore_give %79 : memref<?x?xf32>
                        %113 = loom.subview %arg7[%arg11, %59, %37, %66] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %114 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %114, %113 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %108 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
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
                        %43 = arith.muli %arg12, %c2 : index
                        %44 = arith.addi %arg9, %43 : index
                        %45 = arith.muli %arg11, %c2 : index
                        %46 = arith.muli %arg8, %c4 : index
                        %47 = arith.addi %45, %46 : index
                        %48 = arith.addi %45, %c1 : index
                        %49 = arith.addi %48, %46 : index
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%44, %47], LR : [%44, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %50 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %51 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %52 = loom.semaphore_take %51 : memref<?xf32> -> memref<?xf32>
                        %53 = loom.init_tensor %52[%20] : memref<?xf32> -> tensor<?xf32>
                        %54 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50 : tensor<?xf16>) outs(%53 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %115 = arith.extf %in : f16 to f32
                          linalg.yield %115 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %55 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %56 = loom.semaphore_take %55 : memref<?xf32> -> memref<?xf32>
                        %57 = loom.init_tensor %56[%20] : memref<?xf32> -> tensor<?xf32>
                        %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf32>) outs(%57 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %115 = arith.truncf %cst_0 : f64 to f32
                          %116 = arith.mulf %in, %115 : f32
                          %117 = math.powf %cst, %116 : f32
                          linalg.yield %117 : f32
                        } -> tensor<?xf32>
                        %59 = arith.muli %38, %c256 : index
                        %60 = arith.divui %37, %c16 : index
                        %61 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %62 = loom.semaphore_take %61 : memref<?x16xf16> -> memref<?x16xf16>
                        %63 = loom.subview %arg4[%arg11, %59, %60, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %64 = arith.addi %43, %c1 : index
                        loom.copy %63, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %65 = loom.bufferize_to_tensor %62[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %66 = arith.muli %35, %21 : index
                        %67 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %68 = loom.semaphore_take %67 : memref<?x16xf16> -> memref<?x16xf16>
                        %69 = loom.subview %arg5[%arg11, %38, %37, %66, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %70 = arith.addi %arg10, %45 : index
                        %71 = arith.addi %70, %46 : index
                        loom.copy %69, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %72 = loom.bufferize_to_tensor %68[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %73 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %74 = loom.semaphore_take %73 : memref<16x?xf16> -> memref<16x?xf16>
                        %75 = loom.init_tensor %74[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%72 : tensor<?x16xf16>) outs(%75 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %68 : memref<?x16xf16>
                        %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %77 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %79 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %81 = linalg.fill ins(%cst_1 : f32) outs(%78 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %82 = linalg.matmul ins(%65, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%81 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %74 : memref<16x?xf16>
                        loom.semaphore_give %62 : memref<?x16xf16>
                        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %58 : tensor<?x?xf32>, tensor<?xf32>) outs(%80 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %115 = arith.mulf %in, %in_2 : f32
                          linalg.yield %115 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %77 : memref<?x?xf32>
                        loom.semaphore_give %56 : memref<?xf32>
                        %84 = arith.addi %34, %c1 : index
                        %85 = arith.muli %84, %20 : index
                        %86 = arith.ceildivui %85, %22 : index
                        %87 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                        %89 = loom.init_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %90 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %91 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %92 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %93 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %94 = loom.semaphore_take %93 : memref<?xf32> -> memref<?xf32>
                        %95 = loom.init_tensor %94[%22] : memref<?xf32> -> tensor<?xf32>
                        %96 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %97 = loom.semaphore_take %96 : memref<?xf32> -> memref<?xf32>
                        %98 = loom.init_tensor %97[%22] : memref<?xf32> -> tensor<?xf32>
                        %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                        %101 = scf.for %arg17 = %c0 to %86 step %c1 iter_args(%arg18 = %83) -> (tensor<?x?xf32>) {
                          %115 = arith.muli %arg17, %22 : index
                          %116 = loom.subview %arg0[%arg11, %38, %60, %39, %115] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %116, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%44, %47], LR : [%44, %49]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %117 = loom.bufferize_to_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %118 = loom.subview %arg1[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %118, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %119 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                          %120 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%119 : tensor<?xf16>) outs(%95 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %92 : memref<?xf16>
                          %121 = loom.subview %arg2[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %121, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %122 = loom.bufferize_to_tensor %91[%22] : memref<?xf16> -> tensor<?xf16>
                          %123 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%122 : tensor<?xf16>) outs(%98 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %91 : memref<?xf16>
                          %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %54, %120, %123 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%89 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %128 = arith.truncf %cst_0 : f64 to f32
                            %129 = arith.mulf %in_3, %128 : f32
                            %130 = arith.mulf %in_2, %128 : f32
                            %131 = arith.subf %130, %129 : f32
                            %132 = math.powf %cst, %131 : f32
                            %133 = arith.extf %in : f16 to f32
                            %134 = arith.mulf %133, %132 : f32
                            %135 = arith.mulf %134, %in_4 : f32
                            %136 = arith.truncf %135 : f32 to f16
                            linalg.yield %136 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %97 : memref<?xf32>
                          loom.semaphore_give %94 : memref<?xf32>
                          %125 = loom.subview %arg3[%arg11, %59, %37, %66] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %125, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %126 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %127 = linalg.matmul ins(%124, %126 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          loom.semaphore_give %88 : memref<?x?xf16>
                          scf.yield %127 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %52 : memref<?xf32>
                        %102 = loom.alloc [1] on @L1 : memref<f16>
                        %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                        %104 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        %105 = arith.addi %46, %c3 : index
                        loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %46], LR : [%c7, %105]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                        %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                        %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %110 = loom.subview %arg3[%arg11, %59, %37, %66] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %115 = arith.extf %in_3 : f16 to f32
                          %116 = arith.extf %in_2 : f16 to f32
                          %117 = arith.mulf %116, %115 : f32
                          %118 = arith.addf %in, %117 : f32
                          %119 = arith.truncf %118 : f32 to f16
                          linalg.yield %119 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %103 : memref<f16>
                        loom.semaphore_give %79 : memref<?x?xf32>
                        %113 = loom.subview %arg7[%arg11, %59, %37, %66] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %114 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %114, %113 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %108 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                        %43 = arith.muli %arg11, %c4 : index
                        %44 = arith.addi %arg9, %43 : index
                        %45 = arith.muli %arg12, %c2 : index
                        %46 = arith.muli %arg8, %c4 : index
                        %47 = arith.addi %45, %46 : index
                        %48 = arith.addi %45, %c1 : index
                        %49 = arith.addi %48, %46 : index
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%44, %47], LR : [%44, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %50 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %51 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %52 = loom.semaphore_take %51 : memref<?xf32> -> memref<?xf32>
                        %53 = loom.init_tensor %52[%20] : memref<?xf32> -> tensor<?xf32>
                        %54 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50 : tensor<?xf16>) outs(%53 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %115 = arith.extf %in : f16 to f32
                          linalg.yield %115 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %55 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %56 = loom.semaphore_take %55 : memref<?xf32> -> memref<?xf32>
                        %57 = loom.init_tensor %56[%20] : memref<?xf32> -> tensor<?xf32>
                        %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf32>) outs(%57 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %115 = arith.truncf %cst_0 : f64 to f32
                          %116 = arith.mulf %in, %115 : f32
                          %117 = math.powf %cst, %116 : f32
                          linalg.yield %117 : f32
                        } -> tensor<?xf32>
                        %59 = arith.muli %38, %c256 : index
                        %60 = arith.divui %37, %c16 : index
                        %61 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %62 = loom.semaphore_take %61 : memref<?x16xf16> -> memref<?x16xf16>
                        %63 = loom.subview %arg4[%arg11, %59, %60, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %64 = arith.addi %43, %c3 : index
                        loom.copy %63, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %65 = loom.bufferize_to_tensor %62[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %66 = arith.muli %35, %21 : index
                        %67 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %68 = loom.semaphore_take %67 : memref<?x16xf16> -> memref<?x16xf16>
                        %69 = loom.subview %arg5[%arg11, %38, %37, %66, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %70 = arith.addi %arg10, %45 : index
                        %71 = arith.addi %70, %46 : index
                        loom.copy %69, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %72 = loom.bufferize_to_tensor %68[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %73 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %74 = loom.semaphore_take %73 : memref<16x?xf16> -> memref<16x?xf16>
                        %75 = loom.init_tensor %74[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%72 : tensor<?x16xf16>) outs(%75 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %68 : memref<?x16xf16>
                        %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %77 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %79 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %81 = linalg.fill ins(%cst_1 : f32) outs(%78 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %82 = linalg.matmul ins(%65, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%81 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %74 : memref<16x?xf16>
                        loom.semaphore_give %62 : memref<?x16xf16>
                        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %58 : tensor<?x?xf32>, tensor<?xf32>) outs(%80 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %115 = arith.mulf %in, %in_2 : f32
                          linalg.yield %115 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %77 : memref<?x?xf32>
                        loom.semaphore_give %56 : memref<?xf32>
                        %84 = arith.addi %34, %c1 : index
                        %85 = arith.muli %84, %20 : index
                        %86 = arith.ceildivui %85, %22 : index
                        %87 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                        %89 = loom.init_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %90 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %91 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %92 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %93 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %94 = loom.semaphore_take %93 : memref<?xf32> -> memref<?xf32>
                        %95 = loom.init_tensor %94[%22] : memref<?xf32> -> tensor<?xf32>
                        %96 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %97 = loom.semaphore_take %96 : memref<?xf32> -> memref<?xf32>
                        %98 = loom.init_tensor %97[%22] : memref<?xf32> -> tensor<?xf32>
                        %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                        %101 = scf.for %arg17 = %c0 to %86 step %c1 iter_args(%arg18 = %83) -> (tensor<?x?xf32>) {
                          %115 = arith.muli %arg17, %22 : index
                          %116 = loom.subview %arg0[%arg11, %38, %60, %39, %115] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %116, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%44, %47], LR : [%44, %49]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %117 = loom.bufferize_to_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %118 = loom.subview %arg1[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %118, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %119 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                          %120 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%119 : tensor<?xf16>) outs(%95 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %92 : memref<?xf16>
                          %121 = loom.subview %arg2[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %121, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %122 = loom.bufferize_to_tensor %91[%22] : memref<?xf16> -> tensor<?xf16>
                          %123 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%122 : tensor<?xf16>) outs(%98 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %91 : memref<?xf16>
                          %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %54, %120, %123 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%89 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %128 = arith.truncf %cst_0 : f64 to f32
                            %129 = arith.mulf %in_3, %128 : f32
                            %130 = arith.mulf %in_2, %128 : f32
                            %131 = arith.subf %130, %129 : f32
                            %132 = math.powf %cst, %131 : f32
                            %133 = arith.extf %in : f16 to f32
                            %134 = arith.mulf %133, %132 : f32
                            %135 = arith.mulf %134, %in_4 : f32
                            %136 = arith.truncf %135 : f32 to f16
                            linalg.yield %136 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %97 : memref<?xf32>
                          loom.semaphore_give %94 : memref<?xf32>
                          %125 = loom.subview %arg3[%arg11, %59, %37, %66] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %125, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %126 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %127 = linalg.matmul ins(%124, %126 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          loom.semaphore_give %88 : memref<?x?xf16>
                          scf.yield %127 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %52 : memref<?xf32>
                        %102 = loom.alloc [1] on @L1 : memref<f16>
                        %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                        %104 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        %105 = arith.addi %46, %c3 : index
                        loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %46], LR : [%c7, %105]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                        %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                        %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %110 = loom.subview %arg3[%arg11, %59, %37, %66] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %115 = arith.extf %in_3 : f16 to f32
                          %116 = arith.extf %in_2 : f16 to f32
                          %117 = arith.mulf %116, %115 : f32
                          %118 = arith.addf %in, %117 : f32
                          %119 = arith.truncf %118 : f32 to f16
                          linalg.yield %119 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %103 : memref<f16>
                        loom.semaphore_give %79 : memref<?x?xf32>
                        %113 = loom.subview %arg7[%arg11, %59, %37, %66] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %114 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %114, %113 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %108 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                        %43 = arith.muli %arg12, %c4 : index
                        %44 = arith.addi %arg9, %43 : index
                        %45 = arith.muli %arg11, %c2 : index
                        %46 = arith.muli %arg8, %c4 : index
                        %47 = arith.addi %45, %46 : index
                        %48 = arith.addi %45, %c1 : index
                        %49 = arith.addi %48, %46 : index
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%44, %47], LR : [%44, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %50 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %51 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %52 = loom.semaphore_take %51 : memref<?xf32> -> memref<?xf32>
                        %53 = loom.init_tensor %52[%20] : memref<?xf32> -> tensor<?xf32>
                        %54 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%50 : tensor<?xf16>) outs(%53 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %115 = arith.extf %in : f16 to f32
                          linalg.yield %115 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %55 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %56 = loom.semaphore_take %55 : memref<?xf32> -> memref<?xf32>
                        %57 = loom.init_tensor %56[%20] : memref<?xf32> -> tensor<?xf32>
                        %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf32>) outs(%57 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %115 = arith.truncf %cst_0 : f64 to f32
                          %116 = arith.mulf %in, %115 : f32
                          %117 = math.powf %cst, %116 : f32
                          linalg.yield %117 : f32
                        } -> tensor<?xf32>
                        %59 = arith.muli %38, %c256 : index
                        %60 = arith.divui %37, %c16 : index
                        %61 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %62 = loom.semaphore_take %61 : memref<?x16xf16> -> memref<?x16xf16>
                        %63 = loom.subview %arg4[%arg11, %59, %60, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %64 = arith.addi %43, %c3 : index
                        loom.copy %63, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %65 = loom.bufferize_to_tensor %62[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %66 = arith.muli %35, %21 : index
                        %67 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %68 = loom.semaphore_take %67 : memref<?x16xf16> -> memref<?x16xf16>
                        %69 = loom.subview %arg5[%arg11, %38, %37, %66, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %70 = arith.addi %arg10, %45 : index
                        %71 = arith.addi %70, %46 : index
                        loom.copy %69, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %72 = loom.bufferize_to_tensor %68[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %73 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %74 = loom.semaphore_take %73 : memref<16x?xf16> -> memref<16x?xf16>
                        %75 = loom.init_tensor %74[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%72 : tensor<?x16xf16>) outs(%75 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %68 : memref<?x16xf16>
                        %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %77 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %79 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %81 = linalg.fill ins(%cst_1 : f32) outs(%78 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %82 = linalg.matmul ins(%65, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%81 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %74 : memref<16x?xf16>
                        loom.semaphore_give %62 : memref<?x16xf16>
                        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %58 : tensor<?x?xf32>, tensor<?xf32>) outs(%80 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %115 = arith.mulf %in, %in_2 : f32
                          linalg.yield %115 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %77 : memref<?x?xf32>
                        loom.semaphore_give %56 : memref<?xf32>
                        %84 = arith.addi %34, %c1 : index
                        %85 = arith.muli %84, %20 : index
                        %86 = arith.ceildivui %85, %22 : index
                        %87 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                        %89 = loom.init_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %90 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %91 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %92 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %93 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %94 = loom.semaphore_take %93 : memref<?xf32> -> memref<?xf32>
                        %95 = loom.init_tensor %94[%22] : memref<?xf32> -> tensor<?xf32>
                        %96 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %97 = loom.semaphore_take %96 : memref<?xf32> -> memref<?xf32>
                        %98 = loom.init_tensor %97[%22] : memref<?xf32> -> tensor<?xf32>
                        %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                        %101 = scf.for %arg17 = %c0 to %86 step %c1 iter_args(%arg18 = %83) -> (tensor<?x?xf32>) {
                          %115 = arith.muli %arg17, %22 : index
                          %116 = loom.subview %arg0[%arg11, %38, %60, %39, %115] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %116, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%44, %47], LR : [%44, %49]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %117 = loom.bufferize_to_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %118 = loom.subview %arg1[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %118, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %119 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                          %120 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%119 : tensor<?xf16>) outs(%95 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %92 : memref<?xf16>
                          %121 = loom.subview %arg2[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %121, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%43, %47], LR : [%64, %49]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %122 = loom.bufferize_to_tensor %91[%22] : memref<?xf16> -> tensor<?xf16>
                          %123 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%122 : tensor<?xf16>) outs(%98 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %91 : memref<?xf16>
                          %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %54, %120, %123 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%89 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %128 = arith.truncf %cst_0 : f64 to f32
                            %129 = arith.mulf %in_3, %128 : f32
                            %130 = arith.mulf %in_2, %128 : f32
                            %131 = arith.subf %130, %129 : f32
                            %132 = math.powf %cst, %131 : f32
                            %133 = arith.extf %in : f16 to f32
                            %134 = arith.mulf %133, %132 : f32
                            %135 = arith.mulf %134, %in_4 : f32
                            %136 = arith.truncf %135 : f32 to f16
                            linalg.yield %136 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %97 : memref<?xf32>
                          loom.semaphore_give %94 : memref<?xf32>
                          %125 = loom.subview %arg3[%arg11, %59, %37, %66] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %125, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %126 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %127 = linalg.matmul ins(%124, %126 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          loom.semaphore_give %88 : memref<?x?xf16>
                          scf.yield %127 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %52 : memref<?xf32>
                        %102 = loom.alloc [1] on @L1 : memref<f16>
                        %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                        %104 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        %105 = arith.addi %46, %c3 : index
                        loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %46], LR : [%c7, %105]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                        %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                        %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %110 = loom.subview %arg3[%arg11, %59, %37, %66] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %115 = arith.extf %in_3 : f16 to f32
                          %116 = arith.extf %in_2 : f16 to f32
                          %117 = arith.mulf %116, %115 : f32
                          %118 = arith.addf %in, %117 : f32
                          %119 = arith.truncf %118 : f32 to f16
                          linalg.yield %119 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %103 : memref<f16>
                        loom.semaphore_give %79 : memref<?x?xf32>
                        %113 = loom.subview %arg7[%arg11, %59, %37, %66] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %114 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %114, %113 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%43, %71], LR : [%64, %71]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %108 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                        %43 = arith.muli %arg11, %c2 : index
                        %44 = arith.addi %arg9, %43 : index
                        %45 = arith.muli %arg8, %c4 : index
                        %46 = arith.addi %44, %45 : index
                        %47 = arith.muli %arg12, %c2 : index
                        %48 = arith.addi %47, %c1 : index
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%46, %47], LR : [%46, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %49 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %50 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %51 = loom.semaphore_take %50 : memref<?xf32> -> memref<?xf32>
                        %52 = loom.init_tensor %51[%20] : memref<?xf32> -> tensor<?xf32>
                        %53 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%49 : tensor<?xf16>) outs(%52 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %115 = arith.extf %in : f16 to f32
                          linalg.yield %115 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %54 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %55 = loom.semaphore_take %54 : memref<?xf32> -> memref<?xf32>
                        %56 = loom.init_tensor %55[%20] : memref<?xf32> -> tensor<?xf32>
                        %57 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%53 : tensor<?xf32>) outs(%56 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %115 = arith.truncf %cst_0 : f64 to f32
                          %116 = arith.mulf %in, %115 : f32
                          %117 = math.powf %cst, %116 : f32
                          linalg.yield %117 : f32
                        } -> tensor<?xf32>
                        %58 = arith.muli %38, %c256 : index
                        %59 = arith.divui %37, %c16 : index
                        %60 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                        %62 = loom.subview %arg4[%arg11, %58, %59, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %63 = arith.addi %43, %45 : index
                        %64 = arith.addi %43, %c1 : index
                        %65 = arith.addi %64, %45 : index
                        loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %66 = loom.bufferize_to_tensor %61[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %67 = arith.muli %35, %21 : index
                        %68 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %69 = loom.semaphore_take %68 : memref<?x16xf16> -> memref<?x16xf16>
                        %70 = loom.subview %arg5[%arg11, %38, %37, %67, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %71 = arith.addi %arg10, %47 : index
                        loom.copy %70, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %72 = loom.bufferize_to_tensor %69[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %73 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %74 = loom.semaphore_take %73 : memref<16x?xf16> -> memref<16x?xf16>
                        %75 = loom.init_tensor %74[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%72 : tensor<?x16xf16>) outs(%75 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %69 : memref<?x16xf16>
                        %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %77 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %79 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %81 = linalg.fill ins(%cst_1 : f32) outs(%78 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %82 = linalg.matmul ins(%66, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%81 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %74 : memref<16x?xf16>
                        loom.semaphore_give %61 : memref<?x16xf16>
                        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf32>, tensor<?xf32>) outs(%80 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %115 = arith.mulf %in, %in_2 : f32
                          linalg.yield %115 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %77 : memref<?x?xf32>
                        loom.semaphore_give %55 : memref<?xf32>
                        %84 = arith.addi %34, %c1 : index
                        %85 = arith.muli %84, %20 : index
                        %86 = arith.ceildivui %85, %22 : index
                        %87 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                        %89 = loom.init_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %90 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %91 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %92 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %93 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %94 = loom.semaphore_take %93 : memref<?xf32> -> memref<?xf32>
                        %95 = loom.init_tensor %94[%22] : memref<?xf32> -> tensor<?xf32>
                        %96 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %97 = loom.semaphore_take %96 : memref<?xf32> -> memref<?xf32>
                        %98 = loom.init_tensor %97[%22] : memref<?xf32> -> tensor<?xf32>
                        %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                        %101 = scf.for %arg17 = %c0 to %86 step %c1 iter_args(%arg18 = %83) -> (tensor<?x?xf32>) {
                          %115 = arith.muli %arg17, %22 : index
                          %116 = loom.subview %arg0[%arg11, %38, %59, %39, %115] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %116, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%46, %47], LR : [%46, %48]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %117 = loom.bufferize_to_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %118 = loom.subview %arg1[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %118, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %119 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                          %120 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%119 : tensor<?xf16>) outs(%95 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %92 : memref<?xf16>
                          %121 = loom.subview %arg2[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %121, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %122 = loom.bufferize_to_tensor %91[%22] : memref<?xf16> -> tensor<?xf16>
                          %123 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%122 : tensor<?xf16>) outs(%98 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %91 : memref<?xf16>
                          %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %53, %120, %123 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%89 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %128 = arith.truncf %cst_0 : f64 to f32
                            %129 = arith.mulf %in_3, %128 : f32
                            %130 = arith.mulf %in_2, %128 : f32
                            %131 = arith.subf %130, %129 : f32
                            %132 = math.powf %cst, %131 : f32
                            %133 = arith.extf %in : f16 to f32
                            %134 = arith.mulf %133, %132 : f32
                            %135 = arith.mulf %134, %in_4 : f32
                            %136 = arith.truncf %135 : f32 to f16
                            linalg.yield %136 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %97 : memref<?xf32>
                          loom.semaphore_give %94 : memref<?xf32>
                          %125 = loom.subview %arg3[%arg11, %58, %37, %67] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %125, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %126 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %127 = linalg.matmul ins(%124, %126 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          loom.semaphore_give %88 : memref<?x?xf16>
                          scf.yield %127 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %51 : memref<?xf32>
                        %102 = loom.alloc [1] on @L1 : memref<f16>
                        %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                        %104 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        %105 = arith.addi %45, %c3 : index
                        loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%45, %c0], LR : [%105, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                        %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                        %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %110 = loom.subview %arg3[%arg11, %58, %37, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %115 = arith.extf %in_3 : f16 to f32
                          %116 = arith.extf %in_2 : f16 to f32
                          %117 = arith.mulf %116, %115 : f32
                          %118 = arith.addf %in, %117 : f32
                          %119 = arith.truncf %118 : f32 to f16
                          linalg.yield %119 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %103 : memref<f16>
                        loom.semaphore_give %79 : memref<?x?xf32>
                        %113 = loom.subview %arg7[%arg11, %58, %37, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %114 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %114, %113 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %108 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                        %43 = arith.muli %arg12, %c2 : index
                        %44 = arith.addi %arg9, %43 : index
                        %45 = arith.muli %arg8, %c4 : index
                        %46 = arith.addi %44, %45 : index
                        %47 = arith.muli %arg11, %c2 : index
                        %48 = arith.addi %47, %c1 : index
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%46, %47], LR : [%46, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %49 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %50 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %51 = loom.semaphore_take %50 : memref<?xf32> -> memref<?xf32>
                        %52 = loom.init_tensor %51[%20] : memref<?xf32> -> tensor<?xf32>
                        %53 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%49 : tensor<?xf16>) outs(%52 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %115 = arith.extf %in : f16 to f32
                          linalg.yield %115 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %54 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %55 = loom.semaphore_take %54 : memref<?xf32> -> memref<?xf32>
                        %56 = loom.init_tensor %55[%20] : memref<?xf32> -> tensor<?xf32>
                        %57 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%53 : tensor<?xf32>) outs(%56 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %115 = arith.truncf %cst_0 : f64 to f32
                          %116 = arith.mulf %in, %115 : f32
                          %117 = math.powf %cst, %116 : f32
                          linalg.yield %117 : f32
                        } -> tensor<?xf32>
                        %58 = arith.muli %38, %c256 : index
                        %59 = arith.divui %37, %c16 : index
                        %60 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                        %62 = loom.subview %arg4[%arg11, %58, %59, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %63 = arith.addi %43, %45 : index
                        %64 = arith.addi %43, %c1 : index
                        %65 = arith.addi %64, %45 : index
                        loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %66 = loom.bufferize_to_tensor %61[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %67 = arith.muli %35, %21 : index
                        %68 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %69 = loom.semaphore_take %68 : memref<?x16xf16> -> memref<?x16xf16>
                        %70 = loom.subview %arg5[%arg11, %38, %37, %67, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %71 = arith.addi %arg10, %47 : index
                        loom.copy %70, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %72 = loom.bufferize_to_tensor %69[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %73 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %74 = loom.semaphore_take %73 : memref<16x?xf16> -> memref<16x?xf16>
                        %75 = loom.init_tensor %74[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%72 : tensor<?x16xf16>) outs(%75 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %69 : memref<?x16xf16>
                        %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %77 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %79 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %81 = linalg.fill ins(%cst_1 : f32) outs(%78 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %82 = linalg.matmul ins(%66, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%81 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %74 : memref<16x?xf16>
                        loom.semaphore_give %61 : memref<?x16xf16>
                        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf32>, tensor<?xf32>) outs(%80 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %115 = arith.mulf %in, %in_2 : f32
                          linalg.yield %115 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %77 : memref<?x?xf32>
                        loom.semaphore_give %55 : memref<?xf32>
                        %84 = arith.addi %34, %c1 : index
                        %85 = arith.muli %84, %20 : index
                        %86 = arith.ceildivui %85, %22 : index
                        %87 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                        %89 = loom.init_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %90 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %91 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %92 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %93 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %94 = loom.semaphore_take %93 : memref<?xf32> -> memref<?xf32>
                        %95 = loom.init_tensor %94[%22] : memref<?xf32> -> tensor<?xf32>
                        %96 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %97 = loom.semaphore_take %96 : memref<?xf32> -> memref<?xf32>
                        %98 = loom.init_tensor %97[%22] : memref<?xf32> -> tensor<?xf32>
                        %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                        %101 = scf.for %arg17 = %c0 to %86 step %c1 iter_args(%arg18 = %83) -> (tensor<?x?xf32>) {
                          %115 = arith.muli %arg17, %22 : index
                          %116 = loom.subview %arg0[%arg11, %38, %59, %39, %115] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %116, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%46, %47], LR : [%46, %48]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %117 = loom.bufferize_to_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %118 = loom.subview %arg1[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %118, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %119 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                          %120 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%119 : tensor<?xf16>) outs(%95 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %92 : memref<?xf16>
                          %121 = loom.subview %arg2[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %121, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %122 = loom.bufferize_to_tensor %91[%22] : memref<?xf16> -> tensor<?xf16>
                          %123 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%122 : tensor<?xf16>) outs(%98 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %91 : memref<?xf16>
                          %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %53, %120, %123 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%89 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %128 = arith.truncf %cst_0 : f64 to f32
                            %129 = arith.mulf %in_3, %128 : f32
                            %130 = arith.mulf %in_2, %128 : f32
                            %131 = arith.subf %130, %129 : f32
                            %132 = math.powf %cst, %131 : f32
                            %133 = arith.extf %in : f16 to f32
                            %134 = arith.mulf %133, %132 : f32
                            %135 = arith.mulf %134, %in_4 : f32
                            %136 = arith.truncf %135 : f32 to f16
                            linalg.yield %136 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %97 : memref<?xf32>
                          loom.semaphore_give %94 : memref<?xf32>
                          %125 = loom.subview %arg3[%arg11, %58, %37, %67] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %125, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %126 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %127 = linalg.matmul ins(%124, %126 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          loom.semaphore_give %88 : memref<?x?xf16>
                          scf.yield %127 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %51 : memref<?xf32>
                        %102 = loom.alloc [1] on @L1 : memref<f16>
                        %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                        %104 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        %105 = arith.addi %45, %c3 : index
                        loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%45, %c0], LR : [%105, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                        %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                        %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %110 = loom.subview %arg3[%arg11, %58, %37, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %115 = arith.extf %in_3 : f16 to f32
                          %116 = arith.extf %in_2 : f16 to f32
                          %117 = arith.mulf %116, %115 : f32
                          %118 = arith.addf %in, %117 : f32
                          %119 = arith.truncf %118 : f32 to f16
                          linalg.yield %119 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %103 : memref<f16>
                        loom.semaphore_give %79 : memref<?x?xf32>
                        %113 = loom.subview %arg7[%arg11, %58, %37, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %114 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %114, %113 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %108 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                        %43 = arith.muli %arg11, %c2 : index
                        %44 = arith.addi %arg9, %43 : index
                        %45 = arith.muli %arg8, %c4 : index
                        %46 = arith.addi %44, %45 : index
                        %47 = arith.muli %arg12, %c4 : index
                        %48 = arith.addi %47, %c3 : index
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%46, %47], LR : [%46, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %49 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %50 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %51 = loom.semaphore_take %50 : memref<?xf32> -> memref<?xf32>
                        %52 = loom.init_tensor %51[%20] : memref<?xf32> -> tensor<?xf32>
                        %53 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%49 : tensor<?xf16>) outs(%52 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %115 = arith.extf %in : f16 to f32
                          linalg.yield %115 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %54 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %55 = loom.semaphore_take %54 : memref<?xf32> -> memref<?xf32>
                        %56 = loom.init_tensor %55[%20] : memref<?xf32> -> tensor<?xf32>
                        %57 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%53 : tensor<?xf32>) outs(%56 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %115 = arith.truncf %cst_0 : f64 to f32
                          %116 = arith.mulf %in, %115 : f32
                          %117 = math.powf %cst, %116 : f32
                          linalg.yield %117 : f32
                        } -> tensor<?xf32>
                        %58 = arith.muli %38, %c256 : index
                        %59 = arith.divui %37, %c16 : index
                        %60 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                        %62 = loom.subview %arg4[%arg11, %58, %59, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %63 = arith.addi %43, %45 : index
                        %64 = arith.addi %43, %c1 : index
                        %65 = arith.addi %64, %45 : index
                        loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %66 = loom.bufferize_to_tensor %61[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %67 = arith.muli %35, %21 : index
                        %68 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %69 = loom.semaphore_take %68 : memref<?x16xf16> -> memref<?x16xf16>
                        %70 = loom.subview %arg5[%arg11, %38, %37, %67, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %71 = arith.addi %arg10, %47 : index
                        loom.copy %70, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %72 = loom.bufferize_to_tensor %69[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %73 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %74 = loom.semaphore_take %73 : memref<16x?xf16> -> memref<16x?xf16>
                        %75 = loom.init_tensor %74[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%72 : tensor<?x16xf16>) outs(%75 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %69 : memref<?x16xf16>
                        %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %77 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %79 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %81 = linalg.fill ins(%cst_1 : f32) outs(%78 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %82 = linalg.matmul ins(%66, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%81 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %74 : memref<16x?xf16>
                        loom.semaphore_give %61 : memref<?x16xf16>
                        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf32>, tensor<?xf32>) outs(%80 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %115 = arith.mulf %in, %in_2 : f32
                          linalg.yield %115 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %77 : memref<?x?xf32>
                        loom.semaphore_give %55 : memref<?xf32>
                        %84 = arith.addi %34, %c1 : index
                        %85 = arith.muli %84, %20 : index
                        %86 = arith.ceildivui %85, %22 : index
                        %87 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                        %89 = loom.init_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %90 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %91 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %92 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %93 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %94 = loom.semaphore_take %93 : memref<?xf32> -> memref<?xf32>
                        %95 = loom.init_tensor %94[%22] : memref<?xf32> -> tensor<?xf32>
                        %96 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %97 = loom.semaphore_take %96 : memref<?xf32> -> memref<?xf32>
                        %98 = loom.init_tensor %97[%22] : memref<?xf32> -> tensor<?xf32>
                        %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                        %101 = scf.for %arg17 = %c0 to %86 step %c1 iter_args(%arg18 = %83) -> (tensor<?x?xf32>) {
                          %115 = arith.muli %arg17, %22 : index
                          %116 = loom.subview %arg0[%arg11, %38, %59, %39, %115] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %116, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%46, %47], LR : [%46, %48]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %117 = loom.bufferize_to_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %118 = loom.subview %arg1[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %118, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %119 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                          %120 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%119 : tensor<?xf16>) outs(%95 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %92 : memref<?xf16>
                          %121 = loom.subview %arg2[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %121, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %122 = loom.bufferize_to_tensor %91[%22] : memref<?xf16> -> tensor<?xf16>
                          %123 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%122 : tensor<?xf16>) outs(%98 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %91 : memref<?xf16>
                          %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %53, %120, %123 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%89 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %128 = arith.truncf %cst_0 : f64 to f32
                            %129 = arith.mulf %in_3, %128 : f32
                            %130 = arith.mulf %in_2, %128 : f32
                            %131 = arith.subf %130, %129 : f32
                            %132 = math.powf %cst, %131 : f32
                            %133 = arith.extf %in : f16 to f32
                            %134 = arith.mulf %133, %132 : f32
                            %135 = arith.mulf %134, %in_4 : f32
                            %136 = arith.truncf %135 : f32 to f16
                            linalg.yield %136 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %97 : memref<?xf32>
                          loom.semaphore_give %94 : memref<?xf32>
                          %125 = loom.subview %arg3[%arg11, %58, %37, %67] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %125, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %126 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %127 = linalg.matmul ins(%124, %126 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          loom.semaphore_give %88 : memref<?x?xf16>
                          scf.yield %127 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %51 : memref<?xf32>
                        %102 = loom.alloc [1] on @L1 : memref<f16>
                        %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                        %104 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        %105 = arith.addi %45, %c3 : index
                        loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%45, %c0], LR : [%105, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                        %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                        %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %110 = loom.subview %arg3[%arg11, %58, %37, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %115 = arith.extf %in_3 : f16 to f32
                          %116 = arith.extf %in_2 : f16 to f32
                          %117 = arith.mulf %116, %115 : f32
                          %118 = arith.addf %in, %117 : f32
                          %119 = arith.truncf %118 : f32 to f16
                          linalg.yield %119 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %103 : memref<f16>
                        loom.semaphore_give %79 : memref<?x?xf32>
                        %113 = loom.subview %arg7[%arg11, %58, %37, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %114 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %114, %113 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %108 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                        %43 = arith.muli %arg12, %c2 : index
                        %44 = arith.addi %arg9, %43 : index
                        %45 = arith.muli %arg8, %c4 : index
                        %46 = arith.addi %44, %45 : index
                        %47 = arith.muli %arg11, %c4 : index
                        %48 = arith.addi %47, %c3 : index
                        loom.copy %42, %41 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%46, %47], LR : [%46, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                        %49 = loom.bufferize_to_tensor %41[%20] : memref<?xf16> -> tensor<?xf16>
                        %50 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %51 = loom.semaphore_take %50 : memref<?xf32> -> memref<?xf32>
                        %52 = loom.init_tensor %51[%20] : memref<?xf32> -> tensor<?xf32>
                        %53 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%49 : tensor<?xf16>) outs(%52 : tensor<?xf32>) {
                        ^bb0(%in: f16, %out: f32):
                          %115 = arith.extf %in : f16 to f32
                          linalg.yield %115 : f32
                        } -> tensor<?xf32>
                        loom.semaphore_give %41 : memref<?xf16>
                        %54 = loom.alloc [%20] on @L1 : memref<?xf32>
                        %55 = loom.semaphore_take %54 : memref<?xf32> -> memref<?xf32>
                        %56 = loom.init_tensor %55[%20] : memref<?xf32> -> tensor<?xf32>
                        %57 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%53 : tensor<?xf32>) outs(%56 : tensor<?xf32>) {
                        ^bb0(%in: f32, %out: f32):
                          %115 = arith.truncf %cst_0 : f64 to f32
                          %116 = arith.mulf %in, %115 : f32
                          %117 = math.powf %cst, %116 : f32
                          linalg.yield %117 : f32
                        } -> tensor<?xf32>
                        %58 = arith.muli %38, %c256 : index
                        %59 = arith.divui %37, %c16 : index
                        %60 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                        %61 = loom.semaphore_take %60 : memref<?x16xf16> -> memref<?x16xf16>
                        %62 = loom.subview %arg4[%arg11, %58, %59, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %63 = arith.addi %43, %45 : index
                        %64 = arith.addi %43, %c1 : index
                        %65 = arith.addi %64, %45 : index
                        loom.copy %62, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %66 = loom.bufferize_to_tensor %61[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %67 = arith.muli %35, %21 : index
                        %68 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                        %69 = loom.semaphore_take %68 : memref<?x16xf16> -> memref<?x16xf16>
                        %70 = loom.subview %arg5[%arg11, %38, %37, %67, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                        %71 = arith.addi %arg10, %47 : index
                        loom.copy %70, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                        %72 = loom.bufferize_to_tensor %69[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                        %73 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                        %74 = loom.semaphore_take %73 : memref<16x?xf16> -> memref<16x?xf16>
                        %75 = loom.init_tensor %74[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                        %transposed = linalg.transpose ins(%72 : tensor<?x16xf16>) outs(%75 : tensor<16x?xf16>) permutation = [1, 0] 
                        loom.semaphore_give %69 : memref<?x16xf16>
                        %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                        %77 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %79 = loom.semaphore_take %76 : memref<?x?xf32> -> memref<?x?xf32>
                        %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                        %81 = linalg.fill ins(%cst_1 : f32) outs(%78 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        %82 = linalg.matmul ins(%66, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%81 : tensor<?x?xf32>) -> tensor<?x?xf32>
                        loom.semaphore_give %74 : memref<16x?xf16>
                        loom.semaphore_give %61 : memref<?x16xf16>
                        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf32>, tensor<?xf32>) outs(%80 : tensor<?x?xf32>) {
                        ^bb0(%in: f32, %in_2: f32, %out: f32):
                          %115 = arith.mulf %in, %in_2 : f32
                          linalg.yield %115 : f32
                        } -> tensor<?x?xf32>
                        loom.semaphore_give %77 : memref<?x?xf32>
                        loom.semaphore_give %55 : memref<?xf32>
                        %84 = arith.addi %34, %c1 : index
                        %85 = arith.muli %84, %20 : index
                        %86 = arith.ceildivui %85, %22 : index
                        %87 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                        %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                        %89 = loom.init_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                        %90 = loom.alloc [%22] on @L1 : memref<?xf16>
                        %91 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %92 = loom.semaphore_take %90 : memref<?xf16> -> memref<?xf16>
                        %93 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %94 = loom.semaphore_take %93 : memref<?xf32> -> memref<?xf32>
                        %95 = loom.init_tensor %94[%22] : memref<?xf32> -> tensor<?xf32>
                        %96 = loom.alloc [%22] on @L1 : memref<?xf32>
                        %97 = loom.semaphore_take %96 : memref<?xf32> -> memref<?xf32>
                        %98 = loom.init_tensor %97[%22] : memref<?xf32> -> tensor<?xf32>
                        %99 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                        %100 = loom.semaphore_take %99 : memref<?x?xf16> -> memref<?x?xf16>
                        %101 = scf.for %arg17 = %c0 to %86 step %c1 iter_args(%arg18 = %83) -> (tensor<?x?xf32>) {
                          %115 = arith.muli %arg17, %22 : index
                          %116 = loom.subview %arg0[%arg11, %38, %59, %39, %115] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                          loom.copy %116, %88 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%46, %47], LR : [%46, %48]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                          %117 = loom.bufferize_to_tensor %88[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %118 = loom.subview %arg1[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %118, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %119 = loom.bufferize_to_tensor %92[%22] : memref<?xf16> -> tensor<?xf16>
                          %120 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%119 : tensor<?xf16>) outs(%95 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %92 : memref<?xf16>
                          %121 = loom.subview %arg2[%arg11, %37, %38, %115] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %121, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%63, %47], LR : [%65, %48]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %122 = loom.bufferize_to_tensor %91[%22] : memref<?xf16> -> tensor<?xf16>
                          %123 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%122 : tensor<?xf16>) outs(%98 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %128 = arith.extf %in : f16 to f32
                            linalg.yield %128 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %91 : memref<?xf16>
                          %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%117, %53, %120, %123 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%89 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                            %128 = arith.truncf %cst_0 : f64 to f32
                            %129 = arith.mulf %in_3, %128 : f32
                            %130 = arith.mulf %in_2, %128 : f32
                            %131 = arith.subf %130, %129 : f32
                            %132 = math.powf %cst, %131 : f32
                            %133 = arith.extf %in : f16 to f32
                            %134 = arith.mulf %133, %132 : f32
                            %135 = arith.mulf %134, %in_4 : f32
                            %136 = arith.truncf %135 : f32 to f16
                            linalg.yield %136 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %97 : memref<?xf32>
                          loom.semaphore_give %94 : memref<?xf32>
                          %125 = loom.subview %arg3[%arg11, %58, %37, %67] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %125, %100 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %126 = loom.bufferize_to_tensor %100[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %127 = linalg.matmul ins(%124, %126 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg18 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          loom.semaphore_give %88 : memref<?x?xf16>
                          scf.yield %127 : tensor<?x?xf32>
                        } {loom.iter_type = #loom.iter_type<sequential>}
                        loom.semaphore_give %51 : memref<?xf32>
                        %102 = loom.alloc [1] on @L1 : memref<f16>
                        %103 = loom.semaphore_take %102 : memref<f16> -> memref<f16>
                        %104 = loom.subview %arg6[%37] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                        %105 = arith.addi %45, %c3 : index
                        loom.copy %104, %103 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%45, %c0], LR : [%105, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                        %106 = loom.bufferize_to_tensor %103[] : memref<f16> -> tensor<f16>
                        %107 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                        %108 = loom.semaphore_take %107 : memref<?x?xf16> -> memref<?x?xf16>
                        %109 = loom.init_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %110 = loom.subview %arg3[%arg11, %58, %37, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.copy %110, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                        %111 = loom.bufferize_to_tensor %108[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                        %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %111, %106 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%109 : tensor<?x?xf16>) {
                        ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                          %115 = arith.extf %in_3 : f16 to f32
                          %116 = arith.extf %in_2 : f16 to f32
                          %117 = arith.mulf %116, %115 : f32
                          %118 = arith.addf %in, %117 : f32
                          %119 = arith.truncf %118 : f32 to f16
                          linalg.yield %119 : f16
                        } -> tensor<?x?xf16>
                        loom.semaphore_give %103 : memref<f16>
                        loom.semaphore_give %79 : memref<?x?xf32>
                        %113 = loom.subview %arg7[%arg11, %58, %37, %67] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        %114 = loom.bufferize_to_memref %112 : tensor<?x?xf16> -> memref<?x?xf16>
                        loom.copy %114, %113 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%63, %71], LR : [%65, %71]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                        loom.semaphore_give %108 : memref<?x?xf16>
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
