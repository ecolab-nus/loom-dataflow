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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg12, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %57 = loom.semaphore_take %56 : memref<?xf32> -> memref<?xf32>
                          %58 = loom.init_tensor %57[%20] : memref<?xf32> -> tensor<?xf32>
                          %59 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%55 : tensor<?xf16>) outs(%58 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %120 = arith.extf %in : f16 to f32
                            linalg.yield %120 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %46 : memref<?xf16>
                          %60 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %61 = loom.semaphore_take %60 : memref<?xf32> -> memref<?xf32>
                          %62 = loom.init_tensor %61[%20] : memref<?xf32> -> tensor<?xf32>
                          %63 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%59 : tensor<?xf32>) outs(%62 : tensor<?xf32>) {
                          ^bb0(%in: f32, %out: f32):
                            %120 = arith.truncf %cst_0 : f64 to f32
                            %121 = arith.mulf %in, %120 : f32
                            %122 = math.powf %cst, %121 : f32
                            linalg.yield %122 : f32
                          } -> tensor<?xf32>
                          %64 = arith.muli %43, %c256 : index
                          %65 = arith.divui %42, %c16 : index
                          %66 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %67 = loom.semaphore_take %66 : memref<?x16xf16> -> memref<?x16xf16>
                          %68 = loom.subview %arg4[%41, %64, %65, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %69 = arith.addi %48, %c1 : index
                          loom.copy %68, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %70 = loom.bufferize_to_tensor %67[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %71 = arith.muli %38, %21 : index
                          %72 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %73 = loom.semaphore_take %72 : memref<?x16xf16> -> memref<?x16xf16>
                          %74 = loom.subview %arg5[%41, %43, %42, %71, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %75 = arith.addi %arg10, %50 : index
                          %76 = arith.addi %75, %51 : index
                          loom.copy %74, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %77 = loom.bufferize_to_tensor %73[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %78 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %79 = loom.semaphore_take %78 : memref<16x?xf16> -> memref<16x?xf16>
                          %80 = loom.init_tensor %79[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%77 : tensor<?x16xf16>) outs(%80 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %73 : memref<?x16xf16>
                          %81 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                          %82 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %84 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %85 = loom.init_tensor %84[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %86 = linalg.fill ins(%cst_1 : f32) outs(%83 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          %87 = linalg.matmul ins(%70, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%86 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %79 : memref<16x?xf16>
                          loom.semaphore_give %67 : memref<?x16xf16>
                          %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %63 : tensor<?x?xf32>, tensor<?xf32>) outs(%85 : tensor<?x?xf32>) {
                          ^bb0(%in: f32, %in_2: f32, %out: f32):
                            %120 = arith.mulf %in, %in_2 : f32
                            linalg.yield %120 : f32
                          } -> tensor<?x?xf32>
                          loom.semaphore_give %82 : memref<?x?xf32>
                          loom.semaphore_give %61 : memref<?xf32>
                          %89 = arith.addi %37, %c1 : index
                          %90 = arith.muli %89, %20 : index
                          %91 = arith.ceildivui %90, %22 : index
                          %92 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %96 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %97 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %98 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %99 = loom.semaphore_take %98 : memref<?xf32> -> memref<?xf32>
                          %100 = loom.init_tensor %99[%22] : memref<?xf32> -> tensor<?xf32>
                          %101 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %102 = loom.semaphore_take %101 : memref<?xf32> -> memref<?xf32>
                          %103 = loom.init_tensor %102[%22] : memref<?xf32> -> tensor<?xf32>
                          %104 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %105 = loom.semaphore_take %104 : memref<?x?xf16> -> memref<?x?xf16>
                          %106 = scf.for %arg18 = %c0 to %91 step %c1 iter_args(%arg19 = %88) -> (tensor<?x?xf32>) {
                            %120 = arith.muli %arg18, %22 : index
                            %121 = loom.subview %arg0[%41, %43, %65, %44, %120] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %121, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %122 = loom.bufferize_to_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %123 = loom.subview %arg1[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %123, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %124 = loom.bufferize_to_tensor %97[%22] : memref<?xf16> -> tensor<?xf16>
                            %125 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%124 : tensor<?xf16>) outs(%100 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %97 : memref<?xf16>
                            %126 = loom.subview %arg2[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %126, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %127 = loom.bufferize_to_tensor %96[%22] : memref<?xf16> -> tensor<?xf16>
                            %128 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%127 : tensor<?xf16>) outs(%103 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %96 : memref<?xf16>
                            %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122, %59, %125, %128 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%94 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                              %133 = arith.truncf %cst_0 : f64 to f32
                              %134 = arith.mulf %in_3, %133 : f32
                              %135 = arith.mulf %in_2, %133 : f32
                              %136 = arith.subf %135, %134 : f32
                              %137 = math.powf %cst, %136 : f32
                              %138 = arith.extf %in : f16 to f32
                              %139 = arith.mulf %138, %137 : f32
                              %140 = arith.mulf %139, %in_4 : f32
                              %141 = arith.truncf %140 : f32 to f16
                              linalg.yield %141 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %102 : memref<?xf32>
                            loom.semaphore_give %99 : memref<?xf32>
                            %130 = loom.subview %arg3[%41, %64, %42, %71] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %130, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %131 = loom.bufferize_to_tensor %105[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %132 = linalg.matmul ins(%129, %131 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                            loom.semaphore_give %105 : memref<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %132 : tensor<?x?xf32>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %57 : memref<?xf32>
                          %107 = loom.alloc [1] on @L1 : memref<f16>
                          %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                          %109 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          %110 = arith.addi %51, %c3 : index
                          loom.copy %109, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %110]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %111 = loom.bufferize_to_tensor %108[] : memref<f16> -> tensor<f16>
                          %112 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.subview %arg3[%41, %64, %42, %71] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %115, %113 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %116 = loom.bufferize_to_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %116, %111 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%114 : tensor<?x?xf16>) {
                          ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                            %120 = arith.extf %in_3 : f16 to f32
                            %121 = arith.extf %in_2 : f16 to f32
                            %122 = arith.mulf %121, %120 : f32
                            %123 = arith.addf %in, %122 : f32
                            %124 = arith.truncf %123 : f32 to f16
                            linalg.yield %124 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<f16>
                          loom.semaphore_give %84 : memref<?x?xf32>
                          %118 = loom.subview %arg7[%41, %64, %42, %71] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %119 = loom.bufferize_to_memref %117 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %113 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg11, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %57 = loom.semaphore_take %56 : memref<?xf32> -> memref<?xf32>
                          %58 = loom.init_tensor %57[%20] : memref<?xf32> -> tensor<?xf32>
                          %59 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%55 : tensor<?xf16>) outs(%58 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %120 = arith.extf %in : f16 to f32
                            linalg.yield %120 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %46 : memref<?xf16>
                          %60 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %61 = loom.semaphore_take %60 : memref<?xf32> -> memref<?xf32>
                          %62 = loom.init_tensor %61[%20] : memref<?xf32> -> tensor<?xf32>
                          %63 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%59 : tensor<?xf32>) outs(%62 : tensor<?xf32>) {
                          ^bb0(%in: f32, %out: f32):
                            %120 = arith.truncf %cst_0 : f64 to f32
                            %121 = arith.mulf %in, %120 : f32
                            %122 = math.powf %cst, %121 : f32
                            linalg.yield %122 : f32
                          } -> tensor<?xf32>
                          %64 = arith.muli %43, %c256 : index
                          %65 = arith.divui %42, %c16 : index
                          %66 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %67 = loom.semaphore_take %66 : memref<?x16xf16> -> memref<?x16xf16>
                          %68 = loom.subview %arg4[%41, %64, %65, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %69 = arith.addi %48, %c1 : index
                          loom.copy %68, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %70 = loom.bufferize_to_tensor %67[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %71 = arith.muli %38, %21 : index
                          %72 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %73 = loom.semaphore_take %72 : memref<?x16xf16> -> memref<?x16xf16>
                          %74 = loom.subview %arg5[%41, %43, %42, %71, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %75 = arith.addi %arg10, %50 : index
                          %76 = arith.addi %75, %51 : index
                          loom.copy %74, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %77 = loom.bufferize_to_tensor %73[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %78 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %79 = loom.semaphore_take %78 : memref<16x?xf16> -> memref<16x?xf16>
                          %80 = loom.init_tensor %79[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%77 : tensor<?x16xf16>) outs(%80 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %73 : memref<?x16xf16>
                          %81 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                          %82 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %84 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %85 = loom.init_tensor %84[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %86 = linalg.fill ins(%cst_1 : f32) outs(%83 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          %87 = linalg.matmul ins(%70, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%86 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %79 : memref<16x?xf16>
                          loom.semaphore_give %67 : memref<?x16xf16>
                          %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %63 : tensor<?x?xf32>, tensor<?xf32>) outs(%85 : tensor<?x?xf32>) {
                          ^bb0(%in: f32, %in_2: f32, %out: f32):
                            %120 = arith.mulf %in, %in_2 : f32
                            linalg.yield %120 : f32
                          } -> tensor<?x?xf32>
                          loom.semaphore_give %82 : memref<?x?xf32>
                          loom.semaphore_give %61 : memref<?xf32>
                          %89 = arith.addi %37, %c1 : index
                          %90 = arith.muli %89, %20 : index
                          %91 = arith.ceildivui %90, %22 : index
                          %92 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %96 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %97 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %98 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %99 = loom.semaphore_take %98 : memref<?xf32> -> memref<?xf32>
                          %100 = loom.init_tensor %99[%22] : memref<?xf32> -> tensor<?xf32>
                          %101 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %102 = loom.semaphore_take %101 : memref<?xf32> -> memref<?xf32>
                          %103 = loom.init_tensor %102[%22] : memref<?xf32> -> tensor<?xf32>
                          %104 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %105 = loom.semaphore_take %104 : memref<?x?xf16> -> memref<?x?xf16>
                          %106 = scf.for %arg18 = %c0 to %91 step %c1 iter_args(%arg19 = %88) -> (tensor<?x?xf32>) {
                            %120 = arith.muli %arg18, %22 : index
                            %121 = loom.subview %arg0[%41, %43, %65, %44, %120] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %121, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %122 = loom.bufferize_to_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %123 = loom.subview %arg1[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %123, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %124 = loom.bufferize_to_tensor %97[%22] : memref<?xf16> -> tensor<?xf16>
                            %125 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%124 : tensor<?xf16>) outs(%100 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %97 : memref<?xf16>
                            %126 = loom.subview %arg2[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %126, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %127 = loom.bufferize_to_tensor %96[%22] : memref<?xf16> -> tensor<?xf16>
                            %128 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%127 : tensor<?xf16>) outs(%103 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %96 : memref<?xf16>
                            %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122, %59, %125, %128 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%94 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                              %133 = arith.truncf %cst_0 : f64 to f32
                              %134 = arith.mulf %in_3, %133 : f32
                              %135 = arith.mulf %in_2, %133 : f32
                              %136 = arith.subf %135, %134 : f32
                              %137 = math.powf %cst, %136 : f32
                              %138 = arith.extf %in : f16 to f32
                              %139 = arith.mulf %138, %137 : f32
                              %140 = arith.mulf %139, %in_4 : f32
                              %141 = arith.truncf %140 : f32 to f16
                              linalg.yield %141 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %102 : memref<?xf32>
                            loom.semaphore_give %99 : memref<?xf32>
                            %130 = loom.subview %arg3[%41, %64, %42, %71] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %130, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %131 = loom.bufferize_to_tensor %105[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %132 = linalg.matmul ins(%129, %131 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                            loom.semaphore_give %105 : memref<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %132 : tensor<?x?xf32>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %57 : memref<?xf32>
                          %107 = loom.alloc [1] on @L1 : memref<f16>
                          %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                          %109 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          %110 = arith.addi %51, %c3 : index
                          loom.copy %109, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %110]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %111 = loom.bufferize_to_tensor %108[] : memref<f16> -> tensor<f16>
                          %112 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.subview %arg3[%41, %64, %42, %71] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %115, %113 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %116 = loom.bufferize_to_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %116, %111 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%114 : tensor<?x?xf16>) {
                          ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                            %120 = arith.extf %in_3 : f16 to f32
                            %121 = arith.extf %in_2 : f16 to f32
                            %122 = arith.mulf %121, %120 : f32
                            %123 = arith.addf %in, %122 : f32
                            %124 = arith.truncf %123 : f32 to f16
                            linalg.yield %124 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<f16>
                          loom.semaphore_give %84 : memref<?x?xf32>
                          %118 = loom.subview %arg7[%41, %64, %42, %71] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %119 = loom.bufferize_to_memref %117 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %113 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c4 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg12, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %57 = loom.semaphore_take %56 : memref<?xf32> -> memref<?xf32>
                          %58 = loom.init_tensor %57[%20] : memref<?xf32> -> tensor<?xf32>
                          %59 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%55 : tensor<?xf16>) outs(%58 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %120 = arith.extf %in : f16 to f32
                            linalg.yield %120 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %46 : memref<?xf16>
                          %60 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %61 = loom.semaphore_take %60 : memref<?xf32> -> memref<?xf32>
                          %62 = loom.init_tensor %61[%20] : memref<?xf32> -> tensor<?xf32>
                          %63 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%59 : tensor<?xf32>) outs(%62 : tensor<?xf32>) {
                          ^bb0(%in: f32, %out: f32):
                            %120 = arith.truncf %cst_0 : f64 to f32
                            %121 = arith.mulf %in, %120 : f32
                            %122 = math.powf %cst, %121 : f32
                            linalg.yield %122 : f32
                          } -> tensor<?xf32>
                          %64 = arith.muli %43, %c256 : index
                          %65 = arith.divui %42, %c16 : index
                          %66 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %67 = loom.semaphore_take %66 : memref<?x16xf16> -> memref<?x16xf16>
                          %68 = loom.subview %arg4[%41, %64, %65, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %69 = arith.addi %48, %c3 : index
                          loom.copy %68, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %70 = loom.bufferize_to_tensor %67[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %71 = arith.muli %38, %21 : index
                          %72 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %73 = loom.semaphore_take %72 : memref<?x16xf16> -> memref<?x16xf16>
                          %74 = loom.subview %arg5[%41, %43, %42, %71, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %75 = arith.addi %arg10, %50 : index
                          %76 = arith.addi %75, %51 : index
                          loom.copy %74, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %77 = loom.bufferize_to_tensor %73[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %78 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %79 = loom.semaphore_take %78 : memref<16x?xf16> -> memref<16x?xf16>
                          %80 = loom.init_tensor %79[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%77 : tensor<?x16xf16>) outs(%80 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %73 : memref<?x16xf16>
                          %81 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                          %82 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %84 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %85 = loom.init_tensor %84[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %86 = linalg.fill ins(%cst_1 : f32) outs(%83 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          %87 = linalg.matmul ins(%70, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%86 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %79 : memref<16x?xf16>
                          loom.semaphore_give %67 : memref<?x16xf16>
                          %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %63 : tensor<?x?xf32>, tensor<?xf32>) outs(%85 : tensor<?x?xf32>) {
                          ^bb0(%in: f32, %in_2: f32, %out: f32):
                            %120 = arith.mulf %in, %in_2 : f32
                            linalg.yield %120 : f32
                          } -> tensor<?x?xf32>
                          loom.semaphore_give %82 : memref<?x?xf32>
                          loom.semaphore_give %61 : memref<?xf32>
                          %89 = arith.addi %37, %c1 : index
                          %90 = arith.muli %89, %20 : index
                          %91 = arith.ceildivui %90, %22 : index
                          %92 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %96 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %97 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %98 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %99 = loom.semaphore_take %98 : memref<?xf32> -> memref<?xf32>
                          %100 = loom.init_tensor %99[%22] : memref<?xf32> -> tensor<?xf32>
                          %101 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %102 = loom.semaphore_take %101 : memref<?xf32> -> memref<?xf32>
                          %103 = loom.init_tensor %102[%22] : memref<?xf32> -> tensor<?xf32>
                          %104 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %105 = loom.semaphore_take %104 : memref<?x?xf16> -> memref<?x?xf16>
                          %106 = scf.for %arg18 = %c0 to %91 step %c1 iter_args(%arg19 = %88) -> (tensor<?x?xf32>) {
                            %120 = arith.muli %arg18, %22 : index
                            %121 = loom.subview %arg0[%41, %43, %65, %44, %120] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %121, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %122 = loom.bufferize_to_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %123 = loom.subview %arg1[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %123, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %124 = loom.bufferize_to_tensor %97[%22] : memref<?xf16> -> tensor<?xf16>
                            %125 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%124 : tensor<?xf16>) outs(%100 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %97 : memref<?xf16>
                            %126 = loom.subview %arg2[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %126, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %127 = loom.bufferize_to_tensor %96[%22] : memref<?xf16> -> tensor<?xf16>
                            %128 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%127 : tensor<?xf16>) outs(%103 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %96 : memref<?xf16>
                            %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122, %59, %125, %128 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%94 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                              %133 = arith.truncf %cst_0 : f64 to f32
                              %134 = arith.mulf %in_3, %133 : f32
                              %135 = arith.mulf %in_2, %133 : f32
                              %136 = arith.subf %135, %134 : f32
                              %137 = math.powf %cst, %136 : f32
                              %138 = arith.extf %in : f16 to f32
                              %139 = arith.mulf %138, %137 : f32
                              %140 = arith.mulf %139, %in_4 : f32
                              %141 = arith.truncf %140 : f32 to f16
                              linalg.yield %141 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %102 : memref<?xf32>
                            loom.semaphore_give %99 : memref<?xf32>
                            %130 = loom.subview %arg3[%41, %64, %42, %71] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %130, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %131 = loom.bufferize_to_tensor %105[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %132 = linalg.matmul ins(%129, %131 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                            loom.semaphore_give %105 : memref<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %132 : tensor<?x?xf32>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %57 : memref<?xf32>
                          %107 = loom.alloc [1] on @L1 : memref<f16>
                          %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                          %109 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          %110 = arith.addi %51, %c3 : index
                          loom.copy %109, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %110]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %111 = loom.bufferize_to_tensor %108[] : memref<f16> -> tensor<f16>
                          %112 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.subview %arg3[%41, %64, %42, %71] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %115, %113 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %116 = loom.bufferize_to_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %116, %111 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%114 : tensor<?x?xf16>) {
                          ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                            %120 = arith.extf %in_3 : f16 to f32
                            %121 = arith.extf %in_2 : f16 to f32
                            %122 = arith.mulf %121, %120 : f32
                            %123 = arith.addf %in, %122 : f32
                            %124 = arith.truncf %123 : f32 to f16
                            linalg.yield %124 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<f16>
                          loom.semaphore_give %84 : memref<?x?xf32>
                          %118 = loom.subview %arg7[%41, %64, %42, %71] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %119 = loom.bufferize_to_memref %117 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %113 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c4 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg11, %c2 : index
                          %51 = arith.muli %arg8, %c4 : index
                          %52 = arith.addi %50, %51 : index
                          %53 = arith.addi %50, %c1 : index
                          %54 = arith.addi %53, %51 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %55 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %56 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %57 = loom.semaphore_take %56 : memref<?xf32> -> memref<?xf32>
                          %58 = loom.init_tensor %57[%20] : memref<?xf32> -> tensor<?xf32>
                          %59 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%55 : tensor<?xf16>) outs(%58 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %120 = arith.extf %in : f16 to f32
                            linalg.yield %120 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %46 : memref<?xf16>
                          %60 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %61 = loom.semaphore_take %60 : memref<?xf32> -> memref<?xf32>
                          %62 = loom.init_tensor %61[%20] : memref<?xf32> -> tensor<?xf32>
                          %63 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%59 : tensor<?xf32>) outs(%62 : tensor<?xf32>) {
                          ^bb0(%in: f32, %out: f32):
                            %120 = arith.truncf %cst_0 : f64 to f32
                            %121 = arith.mulf %in, %120 : f32
                            %122 = math.powf %cst, %121 : f32
                            linalg.yield %122 : f32
                          } -> tensor<?xf32>
                          %64 = arith.muli %43, %c256 : index
                          %65 = arith.divui %42, %c16 : index
                          %66 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %67 = loom.semaphore_take %66 : memref<?x16xf16> -> memref<?x16xf16>
                          %68 = loom.subview %arg4[%41, %64, %65, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %69 = arith.addi %48, %c3 : index
                          loom.copy %68, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %70 = loom.bufferize_to_tensor %67[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %71 = arith.muli %38, %21 : index
                          %72 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %73 = loom.semaphore_take %72 : memref<?x16xf16> -> memref<?x16xf16>
                          %74 = loom.subview %arg5[%41, %43, %42, %71, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %75 = arith.addi %arg10, %50 : index
                          %76 = arith.addi %75, %51 : index
                          loom.copy %74, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %77 = loom.bufferize_to_tensor %73[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %78 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %79 = loom.semaphore_take %78 : memref<16x?xf16> -> memref<16x?xf16>
                          %80 = loom.init_tensor %79[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%77 : tensor<?x16xf16>) outs(%80 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %73 : memref<?x16xf16>
                          %81 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                          %82 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %84 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %85 = loom.init_tensor %84[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %86 = linalg.fill ins(%cst_1 : f32) outs(%83 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          %87 = linalg.matmul ins(%70, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%86 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %79 : memref<16x?xf16>
                          loom.semaphore_give %67 : memref<?x16xf16>
                          %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %63 : tensor<?x?xf32>, tensor<?xf32>) outs(%85 : tensor<?x?xf32>) {
                          ^bb0(%in: f32, %in_2: f32, %out: f32):
                            %120 = arith.mulf %in, %in_2 : f32
                            linalg.yield %120 : f32
                          } -> tensor<?x?xf32>
                          loom.semaphore_give %82 : memref<?x?xf32>
                          loom.semaphore_give %61 : memref<?xf32>
                          %89 = arith.addi %37, %c1 : index
                          %90 = arith.muli %89, %20 : index
                          %91 = arith.ceildivui %90, %22 : index
                          %92 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %96 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %97 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %98 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %99 = loom.semaphore_take %98 : memref<?xf32> -> memref<?xf32>
                          %100 = loom.init_tensor %99[%22] : memref<?xf32> -> tensor<?xf32>
                          %101 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %102 = loom.semaphore_take %101 : memref<?xf32> -> memref<?xf32>
                          %103 = loom.init_tensor %102[%22] : memref<?xf32> -> tensor<?xf32>
                          %104 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %105 = loom.semaphore_take %104 : memref<?x?xf16> -> memref<?x?xf16>
                          %106 = scf.for %arg18 = %c0 to %91 step %c1 iter_args(%arg19 = %88) -> (tensor<?x?xf32>) {
                            %120 = arith.muli %arg18, %22 : index
                            %121 = loom.subview %arg0[%41, %43, %65, %44, %120] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %121, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%49, %52], LR : [%49, %54]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %122 = loom.bufferize_to_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %123 = loom.subview %arg1[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %123, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %124 = loom.bufferize_to_tensor %97[%22] : memref<?xf16> -> tensor<?xf16>
                            %125 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%124 : tensor<?xf16>) outs(%100 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %97 : memref<?xf16>
                            %126 = loom.subview %arg2[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %126, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%48, %52], LR : [%69, %54]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %127 = loom.bufferize_to_tensor %96[%22] : memref<?xf16> -> tensor<?xf16>
                            %128 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%127 : tensor<?xf16>) outs(%103 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %96 : memref<?xf16>
                            %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122, %59, %125, %128 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%94 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                              %133 = arith.truncf %cst_0 : f64 to f32
                              %134 = arith.mulf %in_3, %133 : f32
                              %135 = arith.mulf %in_2, %133 : f32
                              %136 = arith.subf %135, %134 : f32
                              %137 = math.powf %cst, %136 : f32
                              %138 = arith.extf %in : f16 to f32
                              %139 = arith.mulf %138, %137 : f32
                              %140 = arith.mulf %139, %in_4 : f32
                              %141 = arith.truncf %140 : f32 to f16
                              linalg.yield %141 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %102 : memref<?xf32>
                            loom.semaphore_give %99 : memref<?xf32>
                            %130 = loom.subview %arg3[%41, %64, %42, %71] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %130, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %131 = loom.bufferize_to_tensor %105[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %132 = linalg.matmul ins(%129, %131 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                            loom.semaphore_give %105 : memref<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %132 : tensor<?x?xf32>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %57 : memref<?xf32>
                          %107 = loom.alloc [1] on @L1 : memref<f16>
                          %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                          %109 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          %110 = arith.addi %51, %c3 : index
                          loom.copy %109, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %51], LR : [%c7, %110]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %111 = loom.bufferize_to_tensor %108[] : memref<f16> -> tensor<f16>
                          %112 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.subview %arg3[%41, %64, %42, %71] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %115, %113 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %116 = loom.bufferize_to_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %116, %111 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%114 : tensor<?x?xf16>) {
                          ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                            %120 = arith.extf %in_3 : f16 to f32
                            %121 = arith.extf %in_2 : f16 to f32
                            %122 = arith.mulf %121, %120 : f32
                            %123 = arith.addf %in, %122 : f32
                            %124 = arith.truncf %123 : f32 to f16
                            linalg.yield %124 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<f16>
                          loom.semaphore_give %84 : memref<?x?xf32>
                          %118 = loom.subview %arg7[%41, %64, %42, %71] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %119 = loom.bufferize_to_memref %117 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%48, %76], LR : [%69, %76]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %113 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg12, %c2 : index
                          %53 = arith.addi %52, %c1 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %56 = loom.semaphore_take %55 : memref<?xf32> -> memref<?xf32>
                          %57 = loom.init_tensor %56[%20] : memref<?xf32> -> tensor<?xf32>
                          %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%57 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %120 = arith.extf %in : f16 to f32
                            linalg.yield %120 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %46 : memref<?xf16>
                          %59 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %60 = loom.semaphore_take %59 : memref<?xf32> -> memref<?xf32>
                          %61 = loom.init_tensor %60[%20] : memref<?xf32> -> tensor<?xf32>
                          %62 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%58 : tensor<?xf32>) outs(%61 : tensor<?xf32>) {
                          ^bb0(%in: f32, %out: f32):
                            %120 = arith.truncf %cst_0 : f64 to f32
                            %121 = arith.mulf %in, %120 : f32
                            %122 = math.powf %cst, %121 : f32
                            linalg.yield %122 : f32
                          } -> tensor<?xf32>
                          %63 = arith.muli %43, %c256 : index
                          %64 = arith.divui %42, %c16 : index
                          %65 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %66 = loom.semaphore_take %65 : memref<?x16xf16> -> memref<?x16xf16>
                          %67 = loom.subview %arg4[%41, %63, %64, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %68 = arith.addi %48, %50 : index
                          %69 = arith.addi %48, %c1 : index
                          %70 = arith.addi %69, %50 : index
                          loom.copy %67, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %71 = loom.bufferize_to_tensor %66[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %72 = arith.muli %38, %21 : index
                          %73 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %74 = loom.semaphore_take %73 : memref<?x16xf16> -> memref<?x16xf16>
                          %75 = loom.subview %arg5[%41, %43, %42, %72, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %76 = arith.addi %arg10, %52 : index
                          loom.copy %75, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %77 = loom.bufferize_to_tensor %74[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %78 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %79 = loom.semaphore_take %78 : memref<16x?xf16> -> memref<16x?xf16>
                          %80 = loom.init_tensor %79[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%77 : tensor<?x16xf16>) outs(%80 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %74 : memref<?x16xf16>
                          %81 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                          %82 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %84 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %85 = loom.init_tensor %84[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %86 = linalg.fill ins(%cst_1 : f32) outs(%83 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          %87 = linalg.matmul ins(%71, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%86 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %79 : memref<16x?xf16>
                          loom.semaphore_give %66 : memref<?x16xf16>
                          %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %62 : tensor<?x?xf32>, tensor<?xf32>) outs(%85 : tensor<?x?xf32>) {
                          ^bb0(%in: f32, %in_2: f32, %out: f32):
                            %120 = arith.mulf %in, %in_2 : f32
                            linalg.yield %120 : f32
                          } -> tensor<?x?xf32>
                          loom.semaphore_give %82 : memref<?x?xf32>
                          loom.semaphore_give %60 : memref<?xf32>
                          %89 = arith.addi %37, %c1 : index
                          %90 = arith.muli %89, %20 : index
                          %91 = arith.ceildivui %90, %22 : index
                          %92 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %96 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %97 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %98 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %99 = loom.semaphore_take %98 : memref<?xf32> -> memref<?xf32>
                          %100 = loom.init_tensor %99[%22] : memref<?xf32> -> tensor<?xf32>
                          %101 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %102 = loom.semaphore_take %101 : memref<?xf32> -> memref<?xf32>
                          %103 = loom.init_tensor %102[%22] : memref<?xf32> -> tensor<?xf32>
                          %104 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %105 = loom.semaphore_take %104 : memref<?x?xf16> -> memref<?x?xf16>
                          %106 = scf.for %arg18 = %c0 to %91 step %c1 iter_args(%arg19 = %88) -> (tensor<?x?xf32>) {
                            %120 = arith.muli %arg18, %22 : index
                            %121 = loom.subview %arg0[%41, %43, %64, %44, %120] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %121, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %122 = loom.bufferize_to_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %123 = loom.subview %arg1[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %123, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %124 = loom.bufferize_to_tensor %97[%22] : memref<?xf16> -> tensor<?xf16>
                            %125 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%124 : tensor<?xf16>) outs(%100 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %97 : memref<?xf16>
                            %126 = loom.subview %arg2[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %126, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %127 = loom.bufferize_to_tensor %96[%22] : memref<?xf16> -> tensor<?xf16>
                            %128 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%127 : tensor<?xf16>) outs(%103 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %96 : memref<?xf16>
                            %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122, %58, %125, %128 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%94 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                              %133 = arith.truncf %cst_0 : f64 to f32
                              %134 = arith.mulf %in_3, %133 : f32
                              %135 = arith.mulf %in_2, %133 : f32
                              %136 = arith.subf %135, %134 : f32
                              %137 = math.powf %cst, %136 : f32
                              %138 = arith.extf %in : f16 to f32
                              %139 = arith.mulf %138, %137 : f32
                              %140 = arith.mulf %139, %in_4 : f32
                              %141 = arith.truncf %140 : f32 to f16
                              linalg.yield %141 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %102 : memref<?xf32>
                            loom.semaphore_give %99 : memref<?xf32>
                            %130 = loom.subview %arg3[%41, %63, %42, %72] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %130, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %131 = loom.bufferize_to_tensor %105[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %132 = linalg.matmul ins(%129, %131 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                            loom.semaphore_give %105 : memref<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %132 : tensor<?x?xf32>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %56 : memref<?xf32>
                          %107 = loom.alloc [1] on @L1 : memref<f16>
                          %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                          %109 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          %110 = arith.addi %50, %c3 : index
                          loom.copy %109, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%110, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %111 = loom.bufferize_to_tensor %108[] : memref<f16> -> tensor<f16>
                          %112 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.subview %arg3[%41, %63, %42, %72] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %115, %113 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %116 = loom.bufferize_to_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %116, %111 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%114 : tensor<?x?xf16>) {
                          ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                            %120 = arith.extf %in_3 : f16 to f32
                            %121 = arith.extf %in_2 : f16 to f32
                            %122 = arith.mulf %121, %120 : f32
                            %123 = arith.addf %in, %122 : f32
                            %124 = arith.truncf %123 : f32 to f16
                            linalg.yield %124 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<f16>
                          loom.semaphore_give %84 : memref<?x?xf32>
                          %118 = loom.subview %arg7[%41, %63, %42, %72] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %119 = loom.bufferize_to_memref %117 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %113 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg11, %c2 : index
                          %53 = arith.addi %52, %c1 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %56 = loom.semaphore_take %55 : memref<?xf32> -> memref<?xf32>
                          %57 = loom.init_tensor %56[%20] : memref<?xf32> -> tensor<?xf32>
                          %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%57 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %120 = arith.extf %in : f16 to f32
                            linalg.yield %120 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %46 : memref<?xf16>
                          %59 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %60 = loom.semaphore_take %59 : memref<?xf32> -> memref<?xf32>
                          %61 = loom.init_tensor %60[%20] : memref<?xf32> -> tensor<?xf32>
                          %62 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%58 : tensor<?xf32>) outs(%61 : tensor<?xf32>) {
                          ^bb0(%in: f32, %out: f32):
                            %120 = arith.truncf %cst_0 : f64 to f32
                            %121 = arith.mulf %in, %120 : f32
                            %122 = math.powf %cst, %121 : f32
                            linalg.yield %122 : f32
                          } -> tensor<?xf32>
                          %63 = arith.muli %43, %c256 : index
                          %64 = arith.divui %42, %c16 : index
                          %65 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %66 = loom.semaphore_take %65 : memref<?x16xf16> -> memref<?x16xf16>
                          %67 = loom.subview %arg4[%41, %63, %64, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %68 = arith.addi %48, %50 : index
                          %69 = arith.addi %48, %c1 : index
                          %70 = arith.addi %69, %50 : index
                          loom.copy %67, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %71 = loom.bufferize_to_tensor %66[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %72 = arith.muli %38, %21 : index
                          %73 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %74 = loom.semaphore_take %73 : memref<?x16xf16> -> memref<?x16xf16>
                          %75 = loom.subview %arg5[%41, %43, %42, %72, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %76 = arith.addi %arg10, %52 : index
                          loom.copy %75, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %77 = loom.bufferize_to_tensor %74[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %78 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %79 = loom.semaphore_take %78 : memref<16x?xf16> -> memref<16x?xf16>
                          %80 = loom.init_tensor %79[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%77 : tensor<?x16xf16>) outs(%80 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %74 : memref<?x16xf16>
                          %81 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                          %82 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %84 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %85 = loom.init_tensor %84[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %86 = linalg.fill ins(%cst_1 : f32) outs(%83 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          %87 = linalg.matmul ins(%71, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%86 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %79 : memref<16x?xf16>
                          loom.semaphore_give %66 : memref<?x16xf16>
                          %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %62 : tensor<?x?xf32>, tensor<?xf32>) outs(%85 : tensor<?x?xf32>) {
                          ^bb0(%in: f32, %in_2: f32, %out: f32):
                            %120 = arith.mulf %in, %in_2 : f32
                            linalg.yield %120 : f32
                          } -> tensor<?x?xf32>
                          loom.semaphore_give %82 : memref<?x?xf32>
                          loom.semaphore_give %60 : memref<?xf32>
                          %89 = arith.addi %37, %c1 : index
                          %90 = arith.muli %89, %20 : index
                          %91 = arith.ceildivui %90, %22 : index
                          %92 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %96 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %97 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %98 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %99 = loom.semaphore_take %98 : memref<?xf32> -> memref<?xf32>
                          %100 = loom.init_tensor %99[%22] : memref<?xf32> -> tensor<?xf32>
                          %101 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %102 = loom.semaphore_take %101 : memref<?xf32> -> memref<?xf32>
                          %103 = loom.init_tensor %102[%22] : memref<?xf32> -> tensor<?xf32>
                          %104 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %105 = loom.semaphore_take %104 : memref<?x?xf16> -> memref<?x?xf16>
                          %106 = scf.for %arg18 = %c0 to %91 step %c1 iter_args(%arg19 = %88) -> (tensor<?x?xf32>) {
                            %120 = arith.muli %arg18, %22 : index
                            %121 = loom.subview %arg0[%41, %43, %64, %44, %120] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %121, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %122 = loom.bufferize_to_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %123 = loom.subview %arg1[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %123, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %124 = loom.bufferize_to_tensor %97[%22] : memref<?xf16> -> tensor<?xf16>
                            %125 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%124 : tensor<?xf16>) outs(%100 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %97 : memref<?xf16>
                            %126 = loom.subview %arg2[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %126, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %127 = loom.bufferize_to_tensor %96[%22] : memref<?xf16> -> tensor<?xf16>
                            %128 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%127 : tensor<?xf16>) outs(%103 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %96 : memref<?xf16>
                            %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122, %58, %125, %128 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%94 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                              %133 = arith.truncf %cst_0 : f64 to f32
                              %134 = arith.mulf %in_3, %133 : f32
                              %135 = arith.mulf %in_2, %133 : f32
                              %136 = arith.subf %135, %134 : f32
                              %137 = math.powf %cst, %136 : f32
                              %138 = arith.extf %in : f16 to f32
                              %139 = arith.mulf %138, %137 : f32
                              %140 = arith.mulf %139, %in_4 : f32
                              %141 = arith.truncf %140 : f32 to f16
                              linalg.yield %141 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %102 : memref<?xf32>
                            loom.semaphore_give %99 : memref<?xf32>
                            %130 = loom.subview %arg3[%41, %63, %42, %72] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %130, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %131 = loom.bufferize_to_tensor %105[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %132 = linalg.matmul ins(%129, %131 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                            loom.semaphore_give %105 : memref<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %132 : tensor<?x?xf32>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %56 : memref<?xf32>
                          %107 = loom.alloc [1] on @L1 : memref<f16>
                          %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                          %109 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          %110 = arith.addi %50, %c3 : index
                          loom.copy %109, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%110, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %111 = loom.bufferize_to_tensor %108[] : memref<f16> -> tensor<f16>
                          %112 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.subview %arg3[%41, %63, %42, %72] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %115, %113 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %116 = loom.bufferize_to_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %116, %111 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%114 : tensor<?x?xf16>) {
                          ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                            %120 = arith.extf %in_3 : f16 to f32
                            %121 = arith.extf %in_2 : f16 to f32
                            %122 = arith.mulf %121, %120 : f32
                            %123 = arith.addf %in, %122 : f32
                            %124 = arith.truncf %123 : f32 to f16
                            linalg.yield %124 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<f16>
                          loom.semaphore_give %84 : memref<?x?xf32>
                          %118 = loom.subview %arg7[%41, %63, %42, %72] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %119 = loom.bufferize_to_memref %117 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %113 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg11, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg12, %c4 : index
                          %53 = arith.addi %52, %c3 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %56 = loom.semaphore_take %55 : memref<?xf32> -> memref<?xf32>
                          %57 = loom.init_tensor %56[%20] : memref<?xf32> -> tensor<?xf32>
                          %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%57 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %120 = arith.extf %in : f16 to f32
                            linalg.yield %120 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %46 : memref<?xf16>
                          %59 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %60 = loom.semaphore_take %59 : memref<?xf32> -> memref<?xf32>
                          %61 = loom.init_tensor %60[%20] : memref<?xf32> -> tensor<?xf32>
                          %62 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%58 : tensor<?xf32>) outs(%61 : tensor<?xf32>) {
                          ^bb0(%in: f32, %out: f32):
                            %120 = arith.truncf %cst_0 : f64 to f32
                            %121 = arith.mulf %in, %120 : f32
                            %122 = math.powf %cst, %121 : f32
                            linalg.yield %122 : f32
                          } -> tensor<?xf32>
                          %63 = arith.muli %43, %c256 : index
                          %64 = arith.divui %42, %c16 : index
                          %65 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %66 = loom.semaphore_take %65 : memref<?x16xf16> -> memref<?x16xf16>
                          %67 = loom.subview %arg4[%41, %63, %64, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %68 = arith.addi %48, %50 : index
                          %69 = arith.addi %48, %c1 : index
                          %70 = arith.addi %69, %50 : index
                          loom.copy %67, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %71 = loom.bufferize_to_tensor %66[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %72 = arith.muli %38, %21 : index
                          %73 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %74 = loom.semaphore_take %73 : memref<?x16xf16> -> memref<?x16xf16>
                          %75 = loom.subview %arg5[%41, %43, %42, %72, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %76 = arith.addi %arg10, %52 : index
                          loom.copy %75, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %77 = loom.bufferize_to_tensor %74[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %78 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %79 = loom.semaphore_take %78 : memref<16x?xf16> -> memref<16x?xf16>
                          %80 = loom.init_tensor %79[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%77 : tensor<?x16xf16>) outs(%80 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %74 : memref<?x16xf16>
                          %81 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                          %82 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %84 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %85 = loom.init_tensor %84[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %86 = linalg.fill ins(%cst_1 : f32) outs(%83 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          %87 = linalg.matmul ins(%71, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%86 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %79 : memref<16x?xf16>
                          loom.semaphore_give %66 : memref<?x16xf16>
                          %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %62 : tensor<?x?xf32>, tensor<?xf32>) outs(%85 : tensor<?x?xf32>) {
                          ^bb0(%in: f32, %in_2: f32, %out: f32):
                            %120 = arith.mulf %in, %in_2 : f32
                            linalg.yield %120 : f32
                          } -> tensor<?x?xf32>
                          loom.semaphore_give %82 : memref<?x?xf32>
                          loom.semaphore_give %60 : memref<?xf32>
                          %89 = arith.addi %37, %c1 : index
                          %90 = arith.muli %89, %20 : index
                          %91 = arith.ceildivui %90, %22 : index
                          %92 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %96 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %97 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %98 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %99 = loom.semaphore_take %98 : memref<?xf32> -> memref<?xf32>
                          %100 = loom.init_tensor %99[%22] : memref<?xf32> -> tensor<?xf32>
                          %101 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %102 = loom.semaphore_take %101 : memref<?xf32> -> memref<?xf32>
                          %103 = loom.init_tensor %102[%22] : memref<?xf32> -> tensor<?xf32>
                          %104 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %105 = loom.semaphore_take %104 : memref<?x?xf16> -> memref<?x?xf16>
                          %106 = scf.for %arg18 = %c0 to %91 step %c1 iter_args(%arg19 = %88) -> (tensor<?x?xf32>) {
                            %120 = arith.muli %arg18, %22 : index
                            %121 = loom.subview %arg0[%41, %43, %64, %44, %120] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %121, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %122 = loom.bufferize_to_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %123 = loom.subview %arg1[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %123, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %124 = loom.bufferize_to_tensor %97[%22] : memref<?xf16> -> tensor<?xf16>
                            %125 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%124 : tensor<?xf16>) outs(%100 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %97 : memref<?xf16>
                            %126 = loom.subview %arg2[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %126, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %127 = loom.bufferize_to_tensor %96[%22] : memref<?xf16> -> tensor<?xf16>
                            %128 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%127 : tensor<?xf16>) outs(%103 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %96 : memref<?xf16>
                            %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122, %58, %125, %128 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%94 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                              %133 = arith.truncf %cst_0 : f64 to f32
                              %134 = arith.mulf %in_3, %133 : f32
                              %135 = arith.mulf %in_2, %133 : f32
                              %136 = arith.subf %135, %134 : f32
                              %137 = math.powf %cst, %136 : f32
                              %138 = arith.extf %in : f16 to f32
                              %139 = arith.mulf %138, %137 : f32
                              %140 = arith.mulf %139, %in_4 : f32
                              %141 = arith.truncf %140 : f32 to f16
                              linalg.yield %141 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %102 : memref<?xf32>
                            loom.semaphore_give %99 : memref<?xf32>
                            %130 = loom.subview %arg3[%41, %63, %42, %72] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %130, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %131 = loom.bufferize_to_tensor %105[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %132 = linalg.matmul ins(%129, %131 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                            loom.semaphore_give %105 : memref<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %132 : tensor<?x?xf32>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %56 : memref<?xf32>
                          %107 = loom.alloc [1] on @L1 : memref<f16>
                          %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                          %109 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          %110 = arith.addi %50, %c3 : index
                          loom.copy %109, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%110, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %111 = loom.bufferize_to_tensor %108[] : memref<f16> -> tensor<f16>
                          %112 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.subview %arg3[%41, %63, %42, %72] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %115, %113 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %116 = loom.bufferize_to_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %116, %111 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%114 : tensor<?x?xf16>) {
                          ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                            %120 = arith.extf %in_3 : f16 to f32
                            %121 = arith.extf %in_2 : f16 to f32
                            %122 = arith.mulf %121, %120 : f32
                            %123 = arith.addf %in, %122 : f32
                            %124 = arith.truncf %123 : f32 to f16
                            linalg.yield %124 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<f16>
                          loom.semaphore_give %84 : memref<?x?xf32>
                          %118 = loom.subview %arg7[%41, %63, %42, %72] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %119 = loom.bufferize_to_memref %117 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %113 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
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
                          %47 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %48 = arith.muli %arg12, %c2 : index
                          %49 = arith.addi %arg9, %48 : index
                          %50 = arith.muli %arg8, %c4 : index
                          %51 = arith.addi %49, %50 : index
                          %52 = arith.muli %arg11, %c4 : index
                          %53 = arith.addi %52, %c3 : index
                          loom.copy %47, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %54 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %55 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %56 = loom.semaphore_take %55 : memref<?xf32> -> memref<?xf32>
                          %57 = loom.init_tensor %56[%20] : memref<?xf32> -> tensor<?xf32>
                          %58 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%57 : tensor<?xf32>) {
                          ^bb0(%in: f16, %out: f32):
                            %120 = arith.extf %in : f16 to f32
                            linalg.yield %120 : f32
                          } -> tensor<?xf32>
                          loom.semaphore_give %46 : memref<?xf16>
                          %59 = loom.alloc [%20] on @L1 : memref<?xf32>
                          %60 = loom.semaphore_take %59 : memref<?xf32> -> memref<?xf32>
                          %61 = loom.init_tensor %60[%20] : memref<?xf32> -> tensor<?xf32>
                          %62 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%58 : tensor<?xf32>) outs(%61 : tensor<?xf32>) {
                          ^bb0(%in: f32, %out: f32):
                            %120 = arith.truncf %cst_0 : f64 to f32
                            %121 = arith.mulf %in, %120 : f32
                            %122 = math.powf %cst, %121 : f32
                            linalg.yield %122 : f32
                          } -> tensor<?xf32>
                          %63 = arith.muli %43, %c256 : index
                          %64 = arith.divui %42, %c16 : index
                          %65 = loom.alloc [%20, 16] on @L1 : memref<?x16xf16>
                          %66 = loom.semaphore_take %65 : memref<?x16xf16> -> memref<?x16xf16>
                          %67 = loom.subview %arg4[%41, %63, %64, 0] [1, %20, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %68 = arith.addi %48, %50 : index
                          %69 = arith.addi %48, %c1 : index
                          %70 = arith.addi %69, %50 : index
                          loom.copy %67, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %71 = loom.bufferize_to_tensor %66[%20, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %72 = arith.muli %38, %21 : index
                          %73 = loom.alloc [%21, 16] on @L1 : memref<?x16xf16>
                          %74 = loom.semaphore_take %73 : memref<?x16xf16> -> memref<?x16xf16>
                          %75 = loom.subview %arg5[%41, %43, %42, %72, 0] [1, 1, 1, %21, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                          %76 = arith.addi %arg10, %52 : index
                          loom.copy %75, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
                          %77 = loom.bufferize_to_tensor %74[%21, 16] : memref<?x16xf16> -> tensor<?x16xf16>
                          %78 = loom.alloc [16, %21] on @L1 : memref<16x?xf16>
                          %79 = loom.semaphore_take %78 : memref<16x?xf16> -> memref<16x?xf16>
                          %80 = loom.init_tensor %79[16, %21] : memref<16x?xf16> -> tensor<16x?xf16>
                          %transposed = linalg.transpose ins(%77 : tensor<?x16xf16>) outs(%80 : tensor<16x?xf16>) permutation = [1, 0] 
                          loom.semaphore_give %74 : memref<?x16xf16>
                          %81 = loom.alloc [%20, %21] on @L1 : memref<?x?xf32>
                          %82 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %83 = loom.init_tensor %82[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %84 = loom.semaphore_take %81 : memref<?x?xf32> -> memref<?x?xf32>
                          %85 = loom.init_tensor %84[%20, %21] : memref<?x?xf32> -> tensor<?x?xf32>
                          %86 = linalg.fill ins(%cst_1 : f32) outs(%83 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          %87 = linalg.matmul ins(%71, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%86 : tensor<?x?xf32>) -> tensor<?x?xf32>
                          loom.semaphore_give %79 : memref<16x?xf16>
                          loom.semaphore_give %66 : memref<?x16xf16>
                          %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%87, %62 : tensor<?x?xf32>, tensor<?xf32>) outs(%85 : tensor<?x?xf32>) {
                          ^bb0(%in: f32, %in_2: f32, %out: f32):
                            %120 = arith.mulf %in, %in_2 : f32
                            linalg.yield %120 : f32
                          } -> tensor<?x?xf32>
                          loom.semaphore_give %82 : memref<?x?xf32>
                          loom.semaphore_give %60 : memref<?xf32>
                          %89 = arith.addi %37, %c1 : index
                          %90 = arith.muli %89, %20 : index
                          %91 = arith.ceildivui %90, %22 : index
                          %92 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %93 = loom.semaphore_take %92 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %96 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %97 = loom.semaphore_take %95 : memref<?xf16> -> memref<?xf16>
                          %98 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %99 = loom.semaphore_take %98 : memref<?xf32> -> memref<?xf32>
                          %100 = loom.init_tensor %99[%22] : memref<?xf32> -> tensor<?xf32>
                          %101 = loom.alloc [%22] on @L1 : memref<?xf32>
                          %102 = loom.semaphore_take %101 : memref<?xf32> -> memref<?xf32>
                          %103 = loom.init_tensor %102[%22] : memref<?xf32> -> tensor<?xf32>
                          %104 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %105 = loom.semaphore_take %104 : memref<?x?xf16> -> memref<?x?xf16>
                          %106 = scf.for %arg18 = %c0 to %91 step %c1 iter_args(%arg19 = %88) -> (tensor<?x?xf32>) {
                            %120 = arith.muli %arg18, %22 : index
                            %121 = loom.subview %arg0[%41, %43, %64, %44, %120] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %121, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%51, %52], LR : [%51, %53]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %122 = loom.bufferize_to_tensor %93[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %123 = loom.subview %arg1[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %123, %97 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %124 = loom.bufferize_to_tensor %97[%22] : memref<?xf16> -> tensor<?xf16>
                            %125 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%124 : tensor<?xf16>) outs(%100 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %97 : memref<?xf16>
                            %126 = loom.subview %arg2[%41, %42, %43, %120] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %126, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%68, %52], LR : [%70, %53]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %127 = loom.bufferize_to_tensor %96[%22] : memref<?xf16> -> tensor<?xf16>
                            %128 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%127 : tensor<?xf16>) outs(%103 : tensor<?xf32>) {
                            ^bb0(%in: f16, %out: f32):
                              %133 = arith.extf %in : f16 to f32
                              linalg.yield %133 : f32
                            } -> tensor<?xf32>
                            loom.semaphore_give %96 : memref<?xf16>
                            %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%122, %58, %125, %128 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%94 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
                              %133 = arith.truncf %cst_0 : f64 to f32
                              %134 = arith.mulf %in_3, %133 : f32
                              %135 = arith.mulf %in_2, %133 : f32
                              %136 = arith.subf %135, %134 : f32
                              %137 = math.powf %cst, %136 : f32
                              %138 = arith.extf %in : f16 to f32
                              %139 = arith.mulf %138, %137 : f32
                              %140 = arith.mulf %139, %in_4 : f32
                              %141 = arith.truncf %140 : f32 to f16
                              linalg.yield %141 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %102 : memref<?xf32>
                            loom.semaphore_give %99 : memref<?xf32>
                            %130 = loom.subview %arg3[%41, %63, %42, %72] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = true, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                            loom.copy %130, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                            %131 = loom.bufferize_to_tensor %105[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %132 = linalg.matmul ins(%129, %131 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf32>) -> tensor<?x?xf32>
                            loom.semaphore_give %105 : memref<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %132 : tensor<?x?xf32>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %56 : memref<?xf32>
                          %107 = loom.alloc [1] on @L1 : memref<f16>
                          %108 = loom.semaphore_take %107 : memref<f16> -> memref<f16>
                          %109 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                          %110 = arith.addi %50, %c3 : index
                          loom.copy %109, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%50, %c0], LR : [%110, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %111 = loom.bufferize_to_tensor %108[] : memref<f16> -> tensor<f16>
                          %112 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.subview %arg3[%41, %63, %42, %72] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.copy %115, %113 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
                          %116 = loom.bufferize_to_tensor %113[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%106, %116, %111 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%114 : tensor<?x?xf16>) {
                          ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
                            %120 = arith.extf %in_3 : f16 to f32
                            %121 = arith.extf %in_2 : f16 to f32
                            %122 = arith.mulf %121, %120 : f32
                            %123 = arith.addf %in, %122 : f32
                            %124 = arith.truncf %123 : f32 to f16
                            linalg.yield %124 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %108 : memref<f16>
                          loom.semaphore_give %84 : memref<?x?xf32>
                          %118 = loom.subview %arg7[%41, %63, %42, %72] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          %119 = loom.bufferize_to_memref %117 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%68, %76], LR : [%70, %76]) : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                          loom.semaphore_give %113 : memref<?x?xf16>
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
