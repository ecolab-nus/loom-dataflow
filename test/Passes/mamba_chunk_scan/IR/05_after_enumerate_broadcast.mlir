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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %50 = arith.muli %arg11, %c2 : index
                          %51 = arith.addi %arg9, %50 : index
                          %52 = arith.muli %arg12, %c2 : index
                          %53 = arith.muli %arg8, %c4 : index
                          %54 = arith.addi %52, %53 : index
                          %55 = arith.addi %52, %c1 : index
                          %56 = arith.addi %55, %53 : index
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %54], LR : [%51, %56]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %57 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %58 = loom.sync ins(%57 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %59 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %60 = loom.semaphore_take %59 : memref<?x32xf16> -> memref<?x32xf16>
                          %61 = loom.init_tensor %60[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %62 = loom.semaphore_take %59 : memref<?x32xf16> -> memref<?x32xf16>
                          %63 = loom.init_tensor %62[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %64 = loom.broadcast ins(%58 : tensor<?xf16>) outs(%63 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %65 = arith.muli %43, %c256 : index
                          %66 = arith.addi %44, %65 : index
                          %67 = arith.divui %42, %c64 : index
                          %68 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %69 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %70 = loom.init_tensor %69[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %71 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %72 = loom.subview %arg4[%41, %66, %67, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %72, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %54], LR : [%51, %56]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %71[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.sync ins(%73 : tensor<?x64xf16>) outs(%70 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %71 : memref<?x64xf16>
                          %75 = arith.muli %38, %21 : index
                          %76 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %77 = loom.semaphore_take %76 : memref<64x?xf16> -> memref<64x?xf16>
                          %78 = loom.init_tensor %77[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %79 = loom.semaphore_take %76 : memref<64x?xf16> -> memref<64x?xf16>
                          %80 = loom.subview %arg5[%41, %43, %42, 0, %75] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          %81 = arith.addi %50, %c1 : index
                          %82 = arith.addi %arg10, %52 : index
                          %83 = arith.addi %82, %53 : index
                          loom.copy %80, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%50, %83], LR : [%81, %83]) : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %84 = loom.bufferize_to_tensor %79[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %85 = loom.sync ins(%84 : tensor<64x?xf16>) outs(%78 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %79 : memref<64x?xf16>
                          %86 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %87 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %88 = loom.init_tensor %87[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %89 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = linalg.fill ins(%cst : f16) outs(%88 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %92 = linalg.matmul ins(%74, %85 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<64x?xf16>
                          loom.semaphore_give %69 : memref<?x64xf16>
                          %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %64 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%90 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %142 = math.exp %in_0 : f16
                            %143 = arith.mulf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?x?xf16>
                          loom.semaphore_give %62 : memref<?x32xf16>
                          %94 = arith.addi %37, %c1 : index
                          %95 = arith.muli %94, %20 : index
                          %96 = arith.ceildivui %95, %22 : index
                          %97 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.init_tensor %98[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %100 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = loom.init_tensor %100[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %102 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %103 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %104 = loom.init_tensor %103[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %105 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %106 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %107 = loom.init_tensor %106[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %110 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %111 = loom.init_tensor %110[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %113 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %114 = loom.init_tensor %113[%22] : memref<?xf16> -> tensor<?xf16>
                          %115 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %116 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %117 = loom.semaphore_take %116 : memref<32x?xf16> -> memref<32x?xf16>
                          %118 = loom.init_tensor %117[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %119 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %120 = loom.semaphore_take %119 : memref<32x?xf16> -> memref<32x?xf16>
                          %121 = loom.init_tensor %120[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %122 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %123 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %124 = loom.init_tensor %123[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %125 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %126 = scf.for %arg18 = %c0 to %96 step %c1 iter_args(%arg19 = %93) -> (tensor<?x?xf16>) {
                            %142 = arith.muli %arg18, %22 : index
                            %143 = loom.subview %arg0[%41, %43, %67, %44, %142] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %143, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %144 = loom.bufferize_to_tensor %108[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %145 = loom.sync ins(%144 : tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %108 : memref<?x?xf16>
                            %146 = loom.subview %arg1[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22] : memref<?xf16> -> tensor<?xf16>
                            %148 = loom.sync ins(%147 : tensor<?xf16>) outs(%114 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %115 : memref<?xf16>
                            %149 = loom.broadcast ins(%58 : tensor<?xf16>) outs(%61 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %150 = loom.broadcast ins(%148 : tensor<?xf16>) outs(%118 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?xf16>
                            %151 = loom.subview %arg2[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %151, %112 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %152 = loom.bufferize_to_tensor %112[%22] : memref<?xf16> -> tensor<?xf16>
                            %153 = loom.sync ins(%152 : tensor<?xf16>) outs(%111 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %112 : memref<?xf16>
                            %154 = loom.broadcast ins(%153 : tensor<?xf16>) outs(%121 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<?xf16>
                            %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145, %149, %150, %154 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %163 = arith.subf %in_0, %in_1 : f16
                              %164 = math.exp %163 : f16
                              %165 = arith.mulf %in, %164 : f16
                              %166 = arith.mulf %165, %in_2 : f16
                              linalg.yield %166 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %120 : memref<32x?xf16>
                            loom.semaphore_give %117 : memref<32x?xf16>
                            loom.semaphore_give %60 : memref<?x32xf16>
                            %156 = arith.addi %142, %65 : index
                            %157 = loom.subview %arg3[%41, %156, %42, %75] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %157, %125 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %158 = loom.bufferize_to_tensor %125[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %159 = loom.sync ins(%158 : tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %125 : memref<?x?xf16>
                            %160 = linalg.fill ins(%cst : f16) outs(%104 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %161 = linalg.matmul ins(%155, %159 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%160 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %123 : memref<?x?xf16>
                            loom.semaphore_give %106 : memref<?x?xf16>
                            %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %161 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %163 = arith.addf %in, %in_0 : f16
                              linalg.yield %163 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?x?xf16>
                            scf.yield %162 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %127 = loom.alloc [1] on @L1 : memref<f16>
                          %128 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %129 = loom.init_tensor %128[] : memref<f16> -> tensor<f16>
                          %130 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %131 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %132 = arith.addi %53, %c3 : index
                          loom.copy %131, %130 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %53], LR : [%c7, %132]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %133 = loom.bufferize_to_tensor %130[] : memref<f16> -> tensor<f16>
                          %134 = loom.sync ins(%133 : tensor<f16>) outs(%129 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %130 : memref<f16>
                          %135 = loom.subview %arg3[%41, %66, %42, %75] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %135, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %136 = loom.bufferize_to_tensor %102[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %137 = loom.sync ins(%136 : tensor<?x?xf16>) outs(%101 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %102 : memref<?x?xf16>
                          %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%126, %137, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%101 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %142 = arith.mulf %in_0, %in_1 : f16
                            %143 = arith.addf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %128 : memref<f16>
                          loom.semaphore_give %89 : memref<?x?xf16>
                          %139 = loom.sync ins(%138 : tensor<?x?xf16>) outs(%99 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          %140 = loom.subview %arg7[%41, %66, %42, %75] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %141 = loom.bufferize_to_memref %139 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %141, %140 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %98 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %50 = arith.muli %arg12, %c2 : index
                          %51 = arith.addi %arg9, %50 : index
                          %52 = arith.muli %arg11, %c2 : index
                          %53 = arith.muli %arg8, %c4 : index
                          %54 = arith.addi %52, %53 : index
                          %55 = arith.addi %52, %c1 : index
                          %56 = arith.addi %55, %53 : index
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %54], LR : [%51, %56]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %57 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %58 = loom.sync ins(%57 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %59 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %60 = loom.semaphore_take %59 : memref<?x32xf16> -> memref<?x32xf16>
                          %61 = loom.init_tensor %60[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %62 = loom.semaphore_take %59 : memref<?x32xf16> -> memref<?x32xf16>
                          %63 = loom.init_tensor %62[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %64 = loom.broadcast ins(%58 : tensor<?xf16>) outs(%63 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %65 = arith.muli %43, %c256 : index
                          %66 = arith.addi %44, %65 : index
                          %67 = arith.divui %42, %c64 : index
                          %68 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %69 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %70 = loom.init_tensor %69[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %71 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %72 = loom.subview %arg4[%41, %66, %67, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %72, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %54], LR : [%51, %56]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %71[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.sync ins(%73 : tensor<?x64xf16>) outs(%70 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %71 : memref<?x64xf16>
                          %75 = arith.muli %38, %21 : index
                          %76 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %77 = loom.semaphore_take %76 : memref<64x?xf16> -> memref<64x?xf16>
                          %78 = loom.init_tensor %77[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %79 = loom.semaphore_take %76 : memref<64x?xf16> -> memref<64x?xf16>
                          %80 = loom.subview %arg5[%41, %43, %42, 0, %75] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          %81 = arith.addi %50, %c1 : index
                          %82 = arith.addi %arg10, %52 : index
                          %83 = arith.addi %82, %53 : index
                          loom.copy %80, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%50, %83], LR : [%81, %83]) : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %84 = loom.bufferize_to_tensor %79[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %85 = loom.sync ins(%84 : tensor<64x?xf16>) outs(%78 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %79 : memref<64x?xf16>
                          %86 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %87 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %88 = loom.init_tensor %87[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %89 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = linalg.fill ins(%cst : f16) outs(%88 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %92 = linalg.matmul ins(%74, %85 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<64x?xf16>
                          loom.semaphore_give %69 : memref<?x64xf16>
                          %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %64 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%90 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %142 = math.exp %in_0 : f16
                            %143 = arith.mulf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?x?xf16>
                          loom.semaphore_give %62 : memref<?x32xf16>
                          %94 = arith.addi %37, %c1 : index
                          %95 = arith.muli %94, %20 : index
                          %96 = arith.ceildivui %95, %22 : index
                          %97 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.init_tensor %98[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %100 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = loom.init_tensor %100[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %102 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %103 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %104 = loom.init_tensor %103[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %105 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %106 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %107 = loom.init_tensor %106[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %110 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %111 = loom.init_tensor %110[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %113 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %114 = loom.init_tensor %113[%22] : memref<?xf16> -> tensor<?xf16>
                          %115 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %116 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %117 = loom.semaphore_take %116 : memref<32x?xf16> -> memref<32x?xf16>
                          %118 = loom.init_tensor %117[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %119 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %120 = loom.semaphore_take %119 : memref<32x?xf16> -> memref<32x?xf16>
                          %121 = loom.init_tensor %120[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %122 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %123 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %124 = loom.init_tensor %123[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %125 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %126 = scf.for %arg18 = %c0 to %96 step %c1 iter_args(%arg19 = %93) -> (tensor<?x?xf16>) {
                            %142 = arith.muli %arg18, %22 : index
                            %143 = loom.subview %arg0[%41, %43, %67, %44, %142] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %143, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %144 = loom.bufferize_to_tensor %108[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %145 = loom.sync ins(%144 : tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %108 : memref<?x?xf16>
                            %146 = loom.subview %arg1[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22] : memref<?xf16> -> tensor<?xf16>
                            %148 = loom.sync ins(%147 : tensor<?xf16>) outs(%114 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %115 : memref<?xf16>
                            %149 = loom.broadcast ins(%58 : tensor<?xf16>) outs(%61 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %150 = loom.broadcast ins(%148 : tensor<?xf16>) outs(%118 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?xf16>
                            %151 = loom.subview %arg2[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %151, %112 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %152 = loom.bufferize_to_tensor %112[%22] : memref<?xf16> -> tensor<?xf16>
                            %153 = loom.sync ins(%152 : tensor<?xf16>) outs(%111 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %112 : memref<?xf16>
                            %154 = loom.broadcast ins(%153 : tensor<?xf16>) outs(%121 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<?xf16>
                            %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145, %149, %150, %154 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %163 = arith.subf %in_0, %in_1 : f16
                              %164 = math.exp %163 : f16
                              %165 = arith.mulf %in, %164 : f16
                              %166 = arith.mulf %165, %in_2 : f16
                              linalg.yield %166 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %120 : memref<32x?xf16>
                            loom.semaphore_give %117 : memref<32x?xf16>
                            loom.semaphore_give %60 : memref<?x32xf16>
                            %156 = arith.addi %142, %65 : index
                            %157 = loom.subview %arg3[%41, %156, %42, %75] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %157, %125 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %158 = loom.bufferize_to_tensor %125[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %159 = loom.sync ins(%158 : tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %125 : memref<?x?xf16>
                            %160 = linalg.fill ins(%cst : f16) outs(%104 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %161 = linalg.matmul ins(%155, %159 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%160 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %123 : memref<?x?xf16>
                            loom.semaphore_give %106 : memref<?x?xf16>
                            %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %161 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %163 = arith.addf %in, %in_0 : f16
                              linalg.yield %163 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?x?xf16>
                            scf.yield %162 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %127 = loom.alloc [1] on @L1 : memref<f16>
                          %128 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %129 = loom.init_tensor %128[] : memref<f16> -> tensor<f16>
                          %130 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %131 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %132 = arith.addi %53, %c3 : index
                          loom.copy %131, %130 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %53], LR : [%c7, %132]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %133 = loom.bufferize_to_tensor %130[] : memref<f16> -> tensor<f16>
                          %134 = loom.sync ins(%133 : tensor<f16>) outs(%129 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %130 : memref<f16>
                          %135 = loom.subview %arg3[%41, %66, %42, %75] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %135, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %136 = loom.bufferize_to_tensor %102[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %137 = loom.sync ins(%136 : tensor<?x?xf16>) outs(%101 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %102 : memref<?x?xf16>
                          %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%126, %137, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%101 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %142 = arith.mulf %in_0, %in_1 : f16
                            %143 = arith.addf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %128 : memref<f16>
                          loom.semaphore_give %89 : memref<?x?xf16>
                          %139 = loom.sync ins(%138 : tensor<?x?xf16>) outs(%99 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          %140 = loom.subview %arg7[%41, %66, %42, %75] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %141 = loom.bufferize_to_memref %139 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %141, %140 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %98 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %50 = arith.muli %arg11, %c4 : index
                          %51 = arith.addi %arg9, %50 : index
                          %52 = arith.muli %arg12, %c2 : index
                          %53 = arith.muli %arg8, %c4 : index
                          %54 = arith.addi %52, %53 : index
                          %55 = arith.addi %52, %c1 : index
                          %56 = arith.addi %55, %53 : index
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %54], LR : [%51, %56]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %57 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %58 = loom.sync ins(%57 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %59 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %60 = loom.semaphore_take %59 : memref<?x32xf16> -> memref<?x32xf16>
                          %61 = loom.init_tensor %60[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %62 = loom.semaphore_take %59 : memref<?x32xf16> -> memref<?x32xf16>
                          %63 = loom.init_tensor %62[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %64 = loom.broadcast ins(%58 : tensor<?xf16>) outs(%63 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %65 = arith.muli %43, %c256 : index
                          %66 = arith.addi %44, %65 : index
                          %67 = arith.divui %42, %c64 : index
                          %68 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %69 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %70 = loom.init_tensor %69[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %71 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %72 = loom.subview %arg4[%41, %66, %67, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %72, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %54], LR : [%51, %56]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %71[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.sync ins(%73 : tensor<?x64xf16>) outs(%70 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %71 : memref<?x64xf16>
                          %75 = arith.muli %38, %21 : index
                          %76 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %77 = loom.semaphore_take %76 : memref<64x?xf16> -> memref<64x?xf16>
                          %78 = loom.init_tensor %77[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %79 = loom.semaphore_take %76 : memref<64x?xf16> -> memref<64x?xf16>
                          %80 = loom.subview %arg5[%41, %43, %42, 0, %75] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          %81 = arith.addi %50, %c3 : index
                          %82 = arith.addi %arg10, %52 : index
                          %83 = arith.addi %82, %53 : index
                          loom.copy %80, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%50, %83], LR : [%81, %83]) : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %84 = loom.bufferize_to_tensor %79[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %85 = loom.sync ins(%84 : tensor<64x?xf16>) outs(%78 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %79 : memref<64x?xf16>
                          %86 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %87 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %88 = loom.init_tensor %87[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %89 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = linalg.fill ins(%cst : f16) outs(%88 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %92 = linalg.matmul ins(%74, %85 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<64x?xf16>
                          loom.semaphore_give %69 : memref<?x64xf16>
                          %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %64 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%90 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %142 = math.exp %in_0 : f16
                            %143 = arith.mulf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?x?xf16>
                          loom.semaphore_give %62 : memref<?x32xf16>
                          %94 = arith.addi %37, %c1 : index
                          %95 = arith.muli %94, %20 : index
                          %96 = arith.ceildivui %95, %22 : index
                          %97 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.init_tensor %98[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %100 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = loom.init_tensor %100[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %102 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %103 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %104 = loom.init_tensor %103[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %105 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %106 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %107 = loom.init_tensor %106[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %110 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %111 = loom.init_tensor %110[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %113 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %114 = loom.init_tensor %113[%22] : memref<?xf16> -> tensor<?xf16>
                          %115 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %116 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %117 = loom.semaphore_take %116 : memref<32x?xf16> -> memref<32x?xf16>
                          %118 = loom.init_tensor %117[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %119 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %120 = loom.semaphore_take %119 : memref<32x?xf16> -> memref<32x?xf16>
                          %121 = loom.init_tensor %120[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %122 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %123 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %124 = loom.init_tensor %123[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %125 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %126 = scf.for %arg18 = %c0 to %96 step %c1 iter_args(%arg19 = %93) -> (tensor<?x?xf16>) {
                            %142 = arith.muli %arg18, %22 : index
                            %143 = loom.subview %arg0[%41, %43, %67, %44, %142] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %143, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %144 = loom.bufferize_to_tensor %108[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %145 = loom.sync ins(%144 : tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %108 : memref<?x?xf16>
                            %146 = loom.subview %arg1[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22] : memref<?xf16> -> tensor<?xf16>
                            %148 = loom.sync ins(%147 : tensor<?xf16>) outs(%114 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %115 : memref<?xf16>
                            %149 = loom.broadcast ins(%58 : tensor<?xf16>) outs(%61 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %150 = loom.broadcast ins(%148 : tensor<?xf16>) outs(%118 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?xf16>
                            %151 = loom.subview %arg2[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %151, %112 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %152 = loom.bufferize_to_tensor %112[%22] : memref<?xf16> -> tensor<?xf16>
                            %153 = loom.sync ins(%152 : tensor<?xf16>) outs(%111 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %112 : memref<?xf16>
                            %154 = loom.broadcast ins(%153 : tensor<?xf16>) outs(%121 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<?xf16>
                            %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145, %149, %150, %154 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %163 = arith.subf %in_0, %in_1 : f16
                              %164 = math.exp %163 : f16
                              %165 = arith.mulf %in, %164 : f16
                              %166 = arith.mulf %165, %in_2 : f16
                              linalg.yield %166 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %120 : memref<32x?xf16>
                            loom.semaphore_give %117 : memref<32x?xf16>
                            loom.semaphore_give %60 : memref<?x32xf16>
                            %156 = arith.addi %142, %65 : index
                            %157 = loom.subview %arg3[%41, %156, %42, %75] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %157, %125 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %158 = loom.bufferize_to_tensor %125[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %159 = loom.sync ins(%158 : tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %125 : memref<?x?xf16>
                            %160 = linalg.fill ins(%cst : f16) outs(%104 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %161 = linalg.matmul ins(%155, %159 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%160 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %123 : memref<?x?xf16>
                            loom.semaphore_give %106 : memref<?x?xf16>
                            %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %161 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %163 = arith.addf %in, %in_0 : f16
                              linalg.yield %163 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?x?xf16>
                            scf.yield %162 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %127 = loom.alloc [1] on @L1 : memref<f16>
                          %128 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %129 = loom.init_tensor %128[] : memref<f16> -> tensor<f16>
                          %130 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %131 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %132 = arith.addi %53, %c3 : index
                          loom.copy %131, %130 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %53], LR : [%c7, %132]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %133 = loom.bufferize_to_tensor %130[] : memref<f16> -> tensor<f16>
                          %134 = loom.sync ins(%133 : tensor<f16>) outs(%129 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %130 : memref<f16>
                          %135 = loom.subview %arg3[%41, %66, %42, %75] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %135, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %136 = loom.bufferize_to_tensor %102[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %137 = loom.sync ins(%136 : tensor<?x?xf16>) outs(%101 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %102 : memref<?x?xf16>
                          %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%126, %137, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%101 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %142 = arith.mulf %in_0, %in_1 : f16
                            %143 = arith.addf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %128 : memref<f16>
                          loom.semaphore_give %89 : memref<?x?xf16>
                          %139 = loom.sync ins(%138 : tensor<?x?xf16>) outs(%99 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          %140 = loom.subview %arg7[%41, %66, %42, %75] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %141 = loom.bufferize_to_memref %139 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %141, %140 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %98 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %50 = arith.muli %arg12, %c4 : index
                          %51 = arith.addi %arg9, %50 : index
                          %52 = arith.muli %arg11, %c2 : index
                          %53 = arith.muli %arg8, %c4 : index
                          %54 = arith.addi %52, %53 : index
                          %55 = arith.addi %52, %c1 : index
                          %56 = arith.addi %55, %53 : index
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %54], LR : [%51, %56]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %57 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %58 = loom.sync ins(%57 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %59 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %60 = loom.semaphore_take %59 : memref<?x32xf16> -> memref<?x32xf16>
                          %61 = loom.init_tensor %60[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %62 = loom.semaphore_take %59 : memref<?x32xf16> -> memref<?x32xf16>
                          %63 = loom.init_tensor %62[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %64 = loom.broadcast ins(%58 : tensor<?xf16>) outs(%63 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %65 = arith.muli %43, %c256 : index
                          %66 = arith.addi %44, %65 : index
                          %67 = arith.divui %42, %c64 : index
                          %68 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %69 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %70 = loom.init_tensor %69[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %71 = loom.semaphore_take %68 : memref<?x64xf16> -> memref<?x64xf16>
                          %72 = loom.subview %arg4[%41, %66, %67, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %72, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%51, %54], LR : [%51, %56]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %73 = loom.bufferize_to_tensor %71[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %74 = loom.sync ins(%73 : tensor<?x64xf16>) outs(%70 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %71 : memref<?x64xf16>
                          %75 = arith.muli %38, %21 : index
                          %76 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %77 = loom.semaphore_take %76 : memref<64x?xf16> -> memref<64x?xf16>
                          %78 = loom.init_tensor %77[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %79 = loom.semaphore_take %76 : memref<64x?xf16> -> memref<64x?xf16>
                          %80 = loom.subview %arg5[%41, %43, %42, 0, %75] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          %81 = arith.addi %50, %c3 : index
                          %82 = arith.addi %arg10, %52 : index
                          %83 = arith.addi %82, %53 : index
                          loom.copy %80, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%50, %83], LR : [%81, %83]) : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %84 = loom.bufferize_to_tensor %79[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %85 = loom.sync ins(%84 : tensor<64x?xf16>) outs(%78 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %79 : memref<64x?xf16>
                          %86 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %87 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %88 = loom.init_tensor %87[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %89 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = linalg.fill ins(%cst : f16) outs(%88 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %92 = linalg.matmul ins(%74, %85 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<64x?xf16>
                          loom.semaphore_give %69 : memref<?x64xf16>
                          %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %64 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%90 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %142 = math.exp %in_0 : f16
                            %143 = arith.mulf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?x?xf16>
                          loom.semaphore_give %62 : memref<?x32xf16>
                          %94 = arith.addi %37, %c1 : index
                          %95 = arith.muli %94, %20 : index
                          %96 = arith.ceildivui %95, %22 : index
                          %97 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.init_tensor %98[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %100 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = loom.init_tensor %100[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %102 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %103 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %104 = loom.init_tensor %103[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %105 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %106 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %107 = loom.init_tensor %106[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %110 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %111 = loom.init_tensor %110[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %113 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %114 = loom.init_tensor %113[%22] : memref<?xf16> -> tensor<?xf16>
                          %115 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %116 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %117 = loom.semaphore_take %116 : memref<32x?xf16> -> memref<32x?xf16>
                          %118 = loom.init_tensor %117[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %119 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %120 = loom.semaphore_take %119 : memref<32x?xf16> -> memref<32x?xf16>
                          %121 = loom.init_tensor %120[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %122 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %123 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %124 = loom.init_tensor %123[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %125 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %126 = scf.for %arg18 = %c0 to %96 step %c1 iter_args(%arg19 = %93) -> (tensor<?x?xf16>) {
                            %142 = arith.muli %arg18, %22 : index
                            %143 = loom.subview %arg0[%41, %43, %67, %44, %142] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %143, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %144 = loom.bufferize_to_tensor %108[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %145 = loom.sync ins(%144 : tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %108 : memref<?x?xf16>
                            %146 = loom.subview %arg1[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22] : memref<?xf16> -> tensor<?xf16>
                            %148 = loom.sync ins(%147 : tensor<?xf16>) outs(%114 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %115 : memref<?xf16>
                            %149 = loom.broadcast ins(%58 : tensor<?xf16>) outs(%61 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %150 = loom.broadcast ins(%148 : tensor<?xf16>) outs(%118 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?xf16>
                            %151 = loom.subview %arg2[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %151, %112 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %152 = loom.bufferize_to_tensor %112[%22] : memref<?xf16> -> tensor<?xf16>
                            %153 = loom.sync ins(%152 : tensor<?xf16>) outs(%111 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %112 : memref<?xf16>
                            %154 = loom.broadcast ins(%153 : tensor<?xf16>) outs(%121 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<?xf16>
                            %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145, %149, %150, %154 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %163 = arith.subf %in_0, %in_1 : f16
                              %164 = math.exp %163 : f16
                              %165 = arith.mulf %in, %164 : f16
                              %166 = arith.mulf %165, %in_2 : f16
                              linalg.yield %166 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %120 : memref<32x?xf16>
                            loom.semaphore_give %117 : memref<32x?xf16>
                            loom.semaphore_give %60 : memref<?x32xf16>
                            %156 = arith.addi %142, %65 : index
                            %157 = loom.subview %arg3[%41, %156, %42, %75] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %157, %125 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %158 = loom.bufferize_to_tensor %125[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %159 = loom.sync ins(%158 : tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %125 : memref<?x?xf16>
                            %160 = linalg.fill ins(%cst : f16) outs(%104 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %161 = linalg.matmul ins(%155, %159 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%160 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %123 : memref<?x?xf16>
                            loom.semaphore_give %106 : memref<?x?xf16>
                            %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %161 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %163 = arith.addf %in, %in_0 : f16
                              linalg.yield %163 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?x?xf16>
                            scf.yield %162 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %127 = loom.alloc [1] on @L1 : memref<f16>
                          %128 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %129 = loom.init_tensor %128[] : memref<f16> -> tensor<f16>
                          %130 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %131 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %132 = arith.addi %53, %c3 : index
                          loom.copy %131, %130 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %53], LR : [%c7, %132]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %133 = loom.bufferize_to_tensor %130[] : memref<f16> -> tensor<f16>
                          %134 = loom.sync ins(%133 : tensor<f16>) outs(%129 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %130 : memref<f16>
                          %135 = loom.subview %arg3[%41, %66, %42, %75] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %135, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %136 = loom.bufferize_to_tensor %102[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %137 = loom.sync ins(%136 : tensor<?x?xf16>) outs(%101 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %102 : memref<?x?xf16>
                          %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%126, %137, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%101 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %142 = arith.mulf %in_0, %in_1 : f16
                            %143 = arith.addf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %128 : memref<f16>
                          loom.semaphore_give %89 : memref<?x?xf16>
                          %139 = loom.sync ins(%138 : tensor<?x?xf16>) outs(%99 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          %140 = loom.subview %arg7[%41, %66, %42, %75] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %141 = loom.bufferize_to_memref %139 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %141, %140 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%51, %83], LR : [%51, %83]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %98 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %50 = arith.muli %arg11, %c2 : index
                          %51 = arith.addi %arg9, %50 : index
                          %52 = arith.muli %arg8, %c4 : index
                          %53 = arith.addi %51, %52 : index
                          %54 = arith.muli %arg12, %c2 : index
                          %55 = arith.addi %54, %c1 : index
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%53, %54], LR : [%53, %55]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %56 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %57 = loom.sync ins(%56 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %58 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %59 = loom.semaphore_take %58 : memref<?x32xf16> -> memref<?x32xf16>
                          %60 = loom.init_tensor %59[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %61 = loom.semaphore_take %58 : memref<?x32xf16> -> memref<?x32xf16>
                          %62 = loom.init_tensor %61[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %63 = loom.broadcast ins(%57 : tensor<?xf16>) outs(%62 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %64 = arith.muli %43, %c256 : index
                          %65 = arith.addi %44, %64 : index
                          %66 = arith.divui %42, %c64 : index
                          %67 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %68 = loom.semaphore_take %67 : memref<?x64xf16> -> memref<?x64xf16>
                          %69 = loom.init_tensor %68[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %70 = loom.semaphore_take %67 : memref<?x64xf16> -> memref<?x64xf16>
                          %71 = loom.subview %arg4[%41, %65, %66, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%53, %54], LR : [%53, %55]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %72 = loom.bufferize_to_tensor %70[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %73 = loom.sync ins(%72 : tensor<?x64xf16>) outs(%69 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %70 : memref<?x64xf16>
                          %74 = arith.muli %38, %21 : index
                          %75 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %76 = loom.semaphore_take %75 : memref<64x?xf16> -> memref<64x?xf16>
                          %77 = loom.init_tensor %76[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %78 = loom.semaphore_take %75 : memref<64x?xf16> -> memref<64x?xf16>
                          %79 = loom.subview %arg5[%41, %43, %42, 0, %74] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          %80 = arith.addi %50, %52 : index
                          %81 = arith.addi %50, %c1 : index
                          %82 = arith.addi %81, %52 : index
                          %83 = arith.addi %arg10, %54 : index
                          loom.copy %79, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%80, %83], LR : [%82, %83]) : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %84 = loom.bufferize_to_tensor %78[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %85 = loom.sync ins(%84 : tensor<64x?xf16>) outs(%77 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %78 : memref<64x?xf16>
                          %86 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %87 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %88 = loom.init_tensor %87[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %89 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = linalg.fill ins(%cst : f16) outs(%88 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %92 = linalg.matmul ins(%73, %85 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %76 : memref<64x?xf16>
                          loom.semaphore_give %68 : memref<?x64xf16>
                          %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %63 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%90 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %142 = math.exp %in_0 : f16
                            %143 = arith.mulf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?x?xf16>
                          loom.semaphore_give %61 : memref<?x32xf16>
                          %94 = arith.addi %37, %c1 : index
                          %95 = arith.muli %94, %20 : index
                          %96 = arith.ceildivui %95, %22 : index
                          %97 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.init_tensor %98[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %100 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = loom.init_tensor %100[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %102 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %103 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %104 = loom.init_tensor %103[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %105 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %106 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %107 = loom.init_tensor %106[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %110 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %111 = loom.init_tensor %110[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %113 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %114 = loom.init_tensor %113[%22] : memref<?xf16> -> tensor<?xf16>
                          %115 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %116 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %117 = loom.semaphore_take %116 : memref<32x?xf16> -> memref<32x?xf16>
                          %118 = loom.init_tensor %117[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %119 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %120 = loom.semaphore_take %119 : memref<32x?xf16> -> memref<32x?xf16>
                          %121 = loom.init_tensor %120[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %122 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %123 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %124 = loom.init_tensor %123[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %125 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %126 = scf.for %arg18 = %c0 to %96 step %c1 iter_args(%arg19 = %93) -> (tensor<?x?xf16>) {
                            %142 = arith.muli %arg18, %22 : index
                            %143 = loom.subview %arg0[%41, %43, %66, %44, %142] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %143, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %144 = loom.bufferize_to_tensor %108[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %145 = loom.sync ins(%144 : tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %108 : memref<?x?xf16>
                            %146 = loom.subview %arg1[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22] : memref<?xf16> -> tensor<?xf16>
                            %148 = loom.sync ins(%147 : tensor<?xf16>) outs(%114 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %115 : memref<?xf16>
                            %149 = loom.broadcast ins(%57 : tensor<?xf16>) outs(%60 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %150 = loom.broadcast ins(%148 : tensor<?xf16>) outs(%118 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?xf16>
                            %151 = loom.subview %arg2[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %151, %112 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %152 = loom.bufferize_to_tensor %112[%22] : memref<?xf16> -> tensor<?xf16>
                            %153 = loom.sync ins(%152 : tensor<?xf16>) outs(%111 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %112 : memref<?xf16>
                            %154 = loom.broadcast ins(%153 : tensor<?xf16>) outs(%121 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<?xf16>
                            %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145, %149, %150, %154 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %163 = arith.subf %in_0, %in_1 : f16
                              %164 = math.exp %163 : f16
                              %165 = arith.mulf %in, %164 : f16
                              %166 = arith.mulf %165, %in_2 : f16
                              linalg.yield %166 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %120 : memref<32x?xf16>
                            loom.semaphore_give %117 : memref<32x?xf16>
                            loom.semaphore_give %59 : memref<?x32xf16>
                            %156 = arith.addi %142, %64 : index
                            %157 = loom.subview %arg3[%41, %156, %42, %74] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %157, %125 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %158 = loom.bufferize_to_tensor %125[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %159 = loom.sync ins(%158 : tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %125 : memref<?x?xf16>
                            %160 = linalg.fill ins(%cst : f16) outs(%104 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %161 = linalg.matmul ins(%155, %159 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%160 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %123 : memref<?x?xf16>
                            loom.semaphore_give %106 : memref<?x?xf16>
                            %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %161 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %163 = arith.addf %in, %in_0 : f16
                              linalg.yield %163 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?x?xf16>
                            scf.yield %162 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %127 = loom.alloc [1] on @L1 : memref<f16>
                          %128 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %129 = loom.init_tensor %128[] : memref<f16> -> tensor<f16>
                          %130 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %131 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %132 = arith.addi %52, %c3 : index
                          loom.copy %131, %130 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%52, %c0], LR : [%132, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %133 = loom.bufferize_to_tensor %130[] : memref<f16> -> tensor<f16>
                          %134 = loom.sync ins(%133 : tensor<f16>) outs(%129 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %130 : memref<f16>
                          %135 = loom.subview %arg3[%41, %65, %42, %74] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %135, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %136 = loom.bufferize_to_tensor %102[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %137 = loom.sync ins(%136 : tensor<?x?xf16>) outs(%101 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %102 : memref<?x?xf16>
                          %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%126, %137, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%101 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %142 = arith.mulf %in_0, %in_1 : f16
                            %143 = arith.addf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %128 : memref<f16>
                          loom.semaphore_give %89 : memref<?x?xf16>
                          %139 = loom.sync ins(%138 : tensor<?x?xf16>) outs(%99 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          %140 = loom.subview %arg7[%41, %65, %42, %74] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %141 = loom.bufferize_to_memref %139 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %141, %140 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %98 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %50 = arith.muli %arg12, %c2 : index
                          %51 = arith.addi %arg9, %50 : index
                          %52 = arith.muli %arg8, %c4 : index
                          %53 = arith.addi %51, %52 : index
                          %54 = arith.muli %arg11, %c2 : index
                          %55 = arith.addi %54, %c1 : index
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%53, %54], LR : [%53, %55]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %56 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %57 = loom.sync ins(%56 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %58 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %59 = loom.semaphore_take %58 : memref<?x32xf16> -> memref<?x32xf16>
                          %60 = loom.init_tensor %59[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %61 = loom.semaphore_take %58 : memref<?x32xf16> -> memref<?x32xf16>
                          %62 = loom.init_tensor %61[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %63 = loom.broadcast ins(%57 : tensor<?xf16>) outs(%62 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %64 = arith.muli %43, %c256 : index
                          %65 = arith.addi %44, %64 : index
                          %66 = arith.divui %42, %c64 : index
                          %67 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %68 = loom.semaphore_take %67 : memref<?x64xf16> -> memref<?x64xf16>
                          %69 = loom.init_tensor %68[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %70 = loom.semaphore_take %67 : memref<?x64xf16> -> memref<?x64xf16>
                          %71 = loom.subview %arg4[%41, %65, %66, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%53, %54], LR : [%53, %55]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %72 = loom.bufferize_to_tensor %70[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %73 = loom.sync ins(%72 : tensor<?x64xf16>) outs(%69 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %70 : memref<?x64xf16>
                          %74 = arith.muli %38, %21 : index
                          %75 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %76 = loom.semaphore_take %75 : memref<64x?xf16> -> memref<64x?xf16>
                          %77 = loom.init_tensor %76[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %78 = loom.semaphore_take %75 : memref<64x?xf16> -> memref<64x?xf16>
                          %79 = loom.subview %arg5[%41, %43, %42, 0, %74] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          %80 = arith.addi %50, %52 : index
                          %81 = arith.addi %50, %c1 : index
                          %82 = arith.addi %81, %52 : index
                          %83 = arith.addi %arg10, %54 : index
                          loom.copy %79, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%80, %83], LR : [%82, %83]) : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %84 = loom.bufferize_to_tensor %78[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %85 = loom.sync ins(%84 : tensor<64x?xf16>) outs(%77 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %78 : memref<64x?xf16>
                          %86 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %87 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %88 = loom.init_tensor %87[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %89 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = linalg.fill ins(%cst : f16) outs(%88 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %92 = linalg.matmul ins(%73, %85 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %76 : memref<64x?xf16>
                          loom.semaphore_give %68 : memref<?x64xf16>
                          %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %63 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%90 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %142 = math.exp %in_0 : f16
                            %143 = arith.mulf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?x?xf16>
                          loom.semaphore_give %61 : memref<?x32xf16>
                          %94 = arith.addi %37, %c1 : index
                          %95 = arith.muli %94, %20 : index
                          %96 = arith.ceildivui %95, %22 : index
                          %97 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.init_tensor %98[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %100 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = loom.init_tensor %100[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %102 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %103 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %104 = loom.init_tensor %103[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %105 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %106 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %107 = loom.init_tensor %106[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %110 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %111 = loom.init_tensor %110[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %113 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %114 = loom.init_tensor %113[%22] : memref<?xf16> -> tensor<?xf16>
                          %115 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %116 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %117 = loom.semaphore_take %116 : memref<32x?xf16> -> memref<32x?xf16>
                          %118 = loom.init_tensor %117[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %119 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %120 = loom.semaphore_take %119 : memref<32x?xf16> -> memref<32x?xf16>
                          %121 = loom.init_tensor %120[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %122 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %123 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %124 = loom.init_tensor %123[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %125 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %126 = scf.for %arg18 = %c0 to %96 step %c1 iter_args(%arg19 = %93) -> (tensor<?x?xf16>) {
                            %142 = arith.muli %arg18, %22 : index
                            %143 = loom.subview %arg0[%41, %43, %66, %44, %142] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %143, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %144 = loom.bufferize_to_tensor %108[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %145 = loom.sync ins(%144 : tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %108 : memref<?x?xf16>
                            %146 = loom.subview %arg1[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22] : memref<?xf16> -> tensor<?xf16>
                            %148 = loom.sync ins(%147 : tensor<?xf16>) outs(%114 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %115 : memref<?xf16>
                            %149 = loom.broadcast ins(%57 : tensor<?xf16>) outs(%60 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %150 = loom.broadcast ins(%148 : tensor<?xf16>) outs(%118 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?xf16>
                            %151 = loom.subview %arg2[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %151, %112 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %152 = loom.bufferize_to_tensor %112[%22] : memref<?xf16> -> tensor<?xf16>
                            %153 = loom.sync ins(%152 : tensor<?xf16>) outs(%111 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %112 : memref<?xf16>
                            %154 = loom.broadcast ins(%153 : tensor<?xf16>) outs(%121 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<?xf16>
                            %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145, %149, %150, %154 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %163 = arith.subf %in_0, %in_1 : f16
                              %164 = math.exp %163 : f16
                              %165 = arith.mulf %in, %164 : f16
                              %166 = arith.mulf %165, %in_2 : f16
                              linalg.yield %166 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %120 : memref<32x?xf16>
                            loom.semaphore_give %117 : memref<32x?xf16>
                            loom.semaphore_give %59 : memref<?x32xf16>
                            %156 = arith.addi %142, %64 : index
                            %157 = loom.subview %arg3[%41, %156, %42, %74] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %157, %125 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %158 = loom.bufferize_to_tensor %125[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %159 = loom.sync ins(%158 : tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %125 : memref<?x?xf16>
                            %160 = linalg.fill ins(%cst : f16) outs(%104 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %161 = linalg.matmul ins(%155, %159 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%160 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %123 : memref<?x?xf16>
                            loom.semaphore_give %106 : memref<?x?xf16>
                            %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %161 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %163 = arith.addf %in, %in_0 : f16
                              linalg.yield %163 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?x?xf16>
                            scf.yield %162 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %127 = loom.alloc [1] on @L1 : memref<f16>
                          %128 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %129 = loom.init_tensor %128[] : memref<f16> -> tensor<f16>
                          %130 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %131 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %132 = arith.addi %52, %c3 : index
                          loom.copy %131, %130 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%52, %c0], LR : [%132, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %133 = loom.bufferize_to_tensor %130[] : memref<f16> -> tensor<f16>
                          %134 = loom.sync ins(%133 : tensor<f16>) outs(%129 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %130 : memref<f16>
                          %135 = loom.subview %arg3[%41, %65, %42, %74] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %135, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %136 = loom.bufferize_to_tensor %102[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %137 = loom.sync ins(%136 : tensor<?x?xf16>) outs(%101 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %102 : memref<?x?xf16>
                          %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%126, %137, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%101 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %142 = arith.mulf %in_0, %in_1 : f16
                            %143 = arith.addf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %128 : memref<f16>
                          loom.semaphore_give %89 : memref<?x?xf16>
                          %139 = loom.sync ins(%138 : tensor<?x?xf16>) outs(%99 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          %140 = loom.subview %arg7[%41, %65, %42, %74] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %141 = loom.bufferize_to_memref %139 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %141, %140 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %98 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %50 = arith.muli %arg11, %c2 : index
                          %51 = arith.addi %arg9, %50 : index
                          %52 = arith.muli %arg8, %c4 : index
                          %53 = arith.addi %51, %52 : index
                          %54 = arith.muli %arg12, %c4 : index
                          %55 = arith.addi %54, %c3 : index
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%53, %54], LR : [%53, %55]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %56 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %57 = loom.sync ins(%56 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %58 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %59 = loom.semaphore_take %58 : memref<?x32xf16> -> memref<?x32xf16>
                          %60 = loom.init_tensor %59[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %61 = loom.semaphore_take %58 : memref<?x32xf16> -> memref<?x32xf16>
                          %62 = loom.init_tensor %61[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %63 = loom.broadcast ins(%57 : tensor<?xf16>) outs(%62 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %64 = arith.muli %43, %c256 : index
                          %65 = arith.addi %44, %64 : index
                          %66 = arith.divui %42, %c64 : index
                          %67 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %68 = loom.semaphore_take %67 : memref<?x64xf16> -> memref<?x64xf16>
                          %69 = loom.init_tensor %68[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %70 = loom.semaphore_take %67 : memref<?x64xf16> -> memref<?x64xf16>
                          %71 = loom.subview %arg4[%41, %65, %66, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%53, %54], LR : [%53, %55]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %72 = loom.bufferize_to_tensor %70[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %73 = loom.sync ins(%72 : tensor<?x64xf16>) outs(%69 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %70 : memref<?x64xf16>
                          %74 = arith.muli %38, %21 : index
                          %75 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %76 = loom.semaphore_take %75 : memref<64x?xf16> -> memref<64x?xf16>
                          %77 = loom.init_tensor %76[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %78 = loom.semaphore_take %75 : memref<64x?xf16> -> memref<64x?xf16>
                          %79 = loom.subview %arg5[%41, %43, %42, 0, %74] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          %80 = arith.addi %50, %52 : index
                          %81 = arith.addi %50, %c1 : index
                          %82 = arith.addi %81, %52 : index
                          %83 = arith.addi %arg10, %54 : index
                          loom.copy %79, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%80, %83], LR : [%82, %83]) : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %84 = loom.bufferize_to_tensor %78[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %85 = loom.sync ins(%84 : tensor<64x?xf16>) outs(%77 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %78 : memref<64x?xf16>
                          %86 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %87 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %88 = loom.init_tensor %87[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %89 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = linalg.fill ins(%cst : f16) outs(%88 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %92 = linalg.matmul ins(%73, %85 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %76 : memref<64x?xf16>
                          loom.semaphore_give %68 : memref<?x64xf16>
                          %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %63 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%90 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %142 = math.exp %in_0 : f16
                            %143 = arith.mulf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?x?xf16>
                          loom.semaphore_give %61 : memref<?x32xf16>
                          %94 = arith.addi %37, %c1 : index
                          %95 = arith.muli %94, %20 : index
                          %96 = arith.ceildivui %95, %22 : index
                          %97 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.init_tensor %98[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %100 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = loom.init_tensor %100[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %102 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %103 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %104 = loom.init_tensor %103[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %105 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %106 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %107 = loom.init_tensor %106[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %110 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %111 = loom.init_tensor %110[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %113 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %114 = loom.init_tensor %113[%22] : memref<?xf16> -> tensor<?xf16>
                          %115 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %116 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %117 = loom.semaphore_take %116 : memref<32x?xf16> -> memref<32x?xf16>
                          %118 = loom.init_tensor %117[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %119 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %120 = loom.semaphore_take %119 : memref<32x?xf16> -> memref<32x?xf16>
                          %121 = loom.init_tensor %120[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %122 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %123 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %124 = loom.init_tensor %123[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %125 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %126 = scf.for %arg18 = %c0 to %96 step %c1 iter_args(%arg19 = %93) -> (tensor<?x?xf16>) {
                            %142 = arith.muli %arg18, %22 : index
                            %143 = loom.subview %arg0[%41, %43, %66, %44, %142] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %143, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %144 = loom.bufferize_to_tensor %108[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %145 = loom.sync ins(%144 : tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %108 : memref<?x?xf16>
                            %146 = loom.subview %arg1[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22] : memref<?xf16> -> tensor<?xf16>
                            %148 = loom.sync ins(%147 : tensor<?xf16>) outs(%114 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %115 : memref<?xf16>
                            %149 = loom.broadcast ins(%57 : tensor<?xf16>) outs(%60 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %150 = loom.broadcast ins(%148 : tensor<?xf16>) outs(%118 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?xf16>
                            %151 = loom.subview %arg2[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %151, %112 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %152 = loom.bufferize_to_tensor %112[%22] : memref<?xf16> -> tensor<?xf16>
                            %153 = loom.sync ins(%152 : tensor<?xf16>) outs(%111 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %112 : memref<?xf16>
                            %154 = loom.broadcast ins(%153 : tensor<?xf16>) outs(%121 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<?xf16>
                            %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145, %149, %150, %154 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %163 = arith.subf %in_0, %in_1 : f16
                              %164 = math.exp %163 : f16
                              %165 = arith.mulf %in, %164 : f16
                              %166 = arith.mulf %165, %in_2 : f16
                              linalg.yield %166 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %120 : memref<32x?xf16>
                            loom.semaphore_give %117 : memref<32x?xf16>
                            loom.semaphore_give %59 : memref<?x32xf16>
                            %156 = arith.addi %142, %64 : index
                            %157 = loom.subview %arg3[%41, %156, %42, %74] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %157, %125 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %158 = loom.bufferize_to_tensor %125[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %159 = loom.sync ins(%158 : tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %125 : memref<?x?xf16>
                            %160 = linalg.fill ins(%cst : f16) outs(%104 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %161 = linalg.matmul ins(%155, %159 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%160 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %123 : memref<?x?xf16>
                            loom.semaphore_give %106 : memref<?x?xf16>
                            %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %161 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %163 = arith.addf %in, %in_0 : f16
                              linalg.yield %163 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?x?xf16>
                            scf.yield %162 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %127 = loom.alloc [1] on @L1 : memref<f16>
                          %128 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %129 = loom.init_tensor %128[] : memref<f16> -> tensor<f16>
                          %130 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %131 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %132 = arith.addi %52, %c3 : index
                          loom.copy %131, %130 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%52, %c0], LR : [%132, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %133 = loom.bufferize_to_tensor %130[] : memref<f16> -> tensor<f16>
                          %134 = loom.sync ins(%133 : tensor<f16>) outs(%129 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %130 : memref<f16>
                          %135 = loom.subview %arg3[%41, %65, %42, %74] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %135, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %136 = loom.bufferize_to_tensor %102[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %137 = loom.sync ins(%136 : tensor<?x?xf16>) outs(%101 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %102 : memref<?x?xf16>
                          %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%126, %137, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%101 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %142 = arith.mulf %in_0, %in_1 : f16
                            %143 = arith.addf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %128 : memref<f16>
                          loom.semaphore_give %89 : memref<?x?xf16>
                          %139 = loom.sync ins(%138 : tensor<?x?xf16>) outs(%99 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          %140 = loom.subview %arg7[%41, %65, %42, %74] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %141 = loom.bufferize_to_memref %139 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %141, %140 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %98 : memref<?x?xf16>
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
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          %50 = arith.muli %arg12, %c2 : index
                          %51 = arith.addi %arg9, %50 : index
                          %52 = arith.muli %arg8, %c4 : index
                          %53 = arith.addi %51, %52 : index
                          %54 = arith.muli %arg11, %c4 : index
                          %55 = arith.addi %54, %c3 : index
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%53, %54], LR : [%53, %55]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %56 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %57 = loom.sync ins(%56 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %58 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %59 = loom.semaphore_take %58 : memref<?x32xf16> -> memref<?x32xf16>
                          %60 = loom.init_tensor %59[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %61 = loom.semaphore_take %58 : memref<?x32xf16> -> memref<?x32xf16>
                          %62 = loom.init_tensor %61[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %63 = loom.broadcast ins(%57 : tensor<?xf16>) outs(%62 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %64 = arith.muli %43, %c256 : index
                          %65 = arith.addi %44, %64 : index
                          %66 = arith.divui %42, %c64 : index
                          %67 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %68 = loom.semaphore_take %67 : memref<?x64xf16> -> memref<?x64xf16>
                          %69 = loom.init_tensor %68[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %70 = loom.semaphore_take %67 : memref<?x64xf16> -> memref<?x64xf16>
                          %71 = loom.subview %arg4[%41, %65, %66, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%53, %54], LR : [%53, %55]) : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %72 = loom.bufferize_to_tensor %70[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %73 = loom.sync ins(%72 : tensor<?x64xf16>) outs(%69 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %70 : memref<?x64xf16>
                          %74 = arith.muli %38, %21 : index
                          %75 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %76 = loom.semaphore_take %75 : memref<64x?xf16> -> memref<64x?xf16>
                          %77 = loom.init_tensor %76[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %78 = loom.semaphore_take %75 : memref<64x?xf16> -> memref<64x?xf16>
                          %79 = loom.subview %arg5[%41, %43, %42, 0, %74] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          %80 = arith.addi %50, %52 : index
                          %81 = arith.addi %50, %c1 : index
                          %82 = arith.addi %81, %52 : index
                          %83 = arith.addi %arg10, %54 : index
                          loom.copy %79, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%80, %83], LR : [%82, %83]) : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %84 = loom.bufferize_to_tensor %78[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %85 = loom.sync ins(%84 : tensor<64x?xf16>) outs(%77 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %78 : memref<64x?xf16>
                          %86 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %87 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %88 = loom.init_tensor %87[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %89 = loom.semaphore_take %86 : memref<?x?xf16> -> memref<?x?xf16>
                          %90 = loom.init_tensor %89[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %91 = linalg.fill ins(%cst : f16) outs(%88 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %92 = linalg.matmul ins(%73, %85 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %76 : memref<64x?xf16>
                          loom.semaphore_give %68 : memref<?x64xf16>
                          %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%92, %63 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%90 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %142 = math.exp %in_0 : f16
                            %143 = arith.mulf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %87 : memref<?x?xf16>
                          loom.semaphore_give %61 : memref<?x32xf16>
                          %94 = arith.addi %37, %c1 : index
                          %95 = arith.muli %94, %20 : index
                          %96 = arith.ceildivui %95, %22 : index
                          %97 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %98 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.init_tensor %98[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %100 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %101 = loom.init_tensor %100[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %102 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %103 = loom.semaphore_take %97 : memref<?x?xf16> -> memref<?x?xf16>
                          %104 = loom.init_tensor %103[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %105 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %106 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %107 = loom.init_tensor %106[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %108 = loom.semaphore_take %105 : memref<?x?xf16> -> memref<?x?xf16>
                          %109 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %110 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %111 = loom.init_tensor %110[%22] : memref<?xf16> -> tensor<?xf16>
                          %112 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %113 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %114 = loom.init_tensor %113[%22] : memref<?xf16> -> tensor<?xf16>
                          %115 = loom.semaphore_take %109 : memref<?xf16> -> memref<?xf16>
                          %116 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %117 = loom.semaphore_take %116 : memref<32x?xf16> -> memref<32x?xf16>
                          %118 = loom.init_tensor %117[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %119 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %120 = loom.semaphore_take %119 : memref<32x?xf16> -> memref<32x?xf16>
                          %121 = loom.init_tensor %120[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %122 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %123 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %124 = loom.init_tensor %123[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %125 = loom.semaphore_take %122 : memref<?x?xf16> -> memref<?x?xf16>
                          %126 = scf.for %arg18 = %c0 to %96 step %c1 iter_args(%arg19 = %93) -> (tensor<?x?xf16>) {
                            %142 = arith.muli %arg18, %22 : index
                            %143 = loom.subview %arg0[%41, %43, %66, %44, %142] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %143, %108 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %144 = loom.bufferize_to_tensor %108[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %145 = loom.sync ins(%144 : tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %108 : memref<?x?xf16>
                            %146 = loom.subview %arg1[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22] : memref<?xf16> -> tensor<?xf16>
                            %148 = loom.sync ins(%147 : tensor<?xf16>) outs(%114 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %115 : memref<?xf16>
                            %149 = loom.broadcast ins(%57 : tensor<?xf16>) outs(%60 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %150 = loom.broadcast ins(%148 : tensor<?xf16>) outs(%118 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?xf16>
                            %151 = loom.subview %arg2[%41, %42, %43, %142] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %151, %112 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %152 = loom.bufferize_to_tensor %112[%22] : memref<?xf16> -> tensor<?xf16>
                            %153 = loom.sync ins(%152 : tensor<?xf16>) outs(%111 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %112 : memref<?xf16>
                            %154 = loom.broadcast ins(%153 : tensor<?xf16>) outs(%121 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<?xf16>
                            %155 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%145, %149, %150, %154 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%107 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %163 = arith.subf %in_0, %in_1 : f16
                              %164 = math.exp %163 : f16
                              %165 = arith.mulf %in, %164 : f16
                              %166 = arith.mulf %165, %in_2 : f16
                              linalg.yield %166 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %120 : memref<32x?xf16>
                            loom.semaphore_give %117 : memref<32x?xf16>
                            loom.semaphore_give %59 : memref<?x32xf16>
                            %156 = arith.addi %142, %64 : index
                            %157 = loom.subview %arg3[%41, %156, %42, %74] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %157, %125 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %158 = loom.bufferize_to_tensor %125[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %159 = loom.sync ins(%158 : tensor<?x?xf16>) outs(%124 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %125 : memref<?x?xf16>
                            %160 = linalg.fill ins(%cst : f16) outs(%104 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %161 = linalg.matmul ins(%155, %159 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%160 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %123 : memref<?x?xf16>
                            loom.semaphore_give %106 : memref<?x?xf16>
                            %162 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %161 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %163 = arith.addf %in, %in_0 : f16
                              linalg.yield %163 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?x?xf16>
                            scf.yield %162 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %127 = loom.alloc [1] on @L1 : memref<f16>
                          %128 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %129 = loom.init_tensor %128[] : memref<f16> -> tensor<f16>
                          %130 = loom.semaphore_take %127 : memref<f16> -> memref<f16>
                          %131 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          %132 = arith.addi %52, %c3 : index
                          loom.copy %131, %130 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%52, %c0], LR : [%132, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %133 = loom.bufferize_to_tensor %130[] : memref<f16> -> tensor<f16>
                          %134 = loom.sync ins(%133 : tensor<f16>) outs(%129 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %130 : memref<f16>
                          %135 = loom.subview %arg3[%41, %65, %42, %74] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %135, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %136 = loom.bufferize_to_tensor %102[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %137 = loom.sync ins(%136 : tensor<?x?xf16>) outs(%101 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %102 : memref<?x?xf16>
                          %138 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%126, %137, %134 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%101 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %142 = arith.mulf %in_0, %in_1 : f16
                            %143 = arith.addf %in, %142 : f16
                            linalg.yield %143 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %128 : memref<f16>
                          loom.semaphore_give %89 : memref<?x?xf16>
                          %139 = loom.sync ins(%138 : tensor<?x?xf16>) outs(%99 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %100 : memref<?x?xf16>
                          %140 = loom.subview %arg7[%41, %65, %42, %74] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %141 = loom.bufferize_to_memref %139 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %141, %140 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%53, %83], LR : [%53, %83]) : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %98 : memref<?x?xf16>
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
