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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %50 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %51 = loom.sync ins(%50 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %52 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %53 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %54 = loom.init_tensor %53[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %55 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %56 = loom.init_tensor %55[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %57 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%56 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %58 = arith.muli %43, %c256 : index
                          %59 = arith.addi %44, %58 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.init_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %64 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %65 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %64[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = loom.sync ins(%66 : tensor<?x64xf16>) outs(%63 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %64 : memref<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %70 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %71 = loom.init_tensor %70[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %72 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %73 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %73, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %74 = loom.bufferize_to_tensor %72[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %75 = loom.sync ins(%74 : tensor<64x?xf16>) outs(%71 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %72 : memref<64x?xf16>
                          %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %77 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %79 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = linalg.fill ins(%cst : f16) outs(%78 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %82 = linalg.matmul ins(%67, %75 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%80 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %131 = math.exp %in_0 : f16
                            %132 = arith.mulf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<?x?xf16>
                          loom.semaphore_give %55 : memref<?x32xf16>
                          %84 = arith.addi %37, %c1 : index
                          %85 = arith.muli %84, %20 : index
                          %86 = arith.ceildivui %85, %22 : index
                          %87 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %89 = loom.init_tensor %88[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %90 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %91 = loom.init_tensor %90[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %92 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %93 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = loom.init_tensor %96[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %98 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %100 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %101 = loom.init_tensor %100[%22] : memref<?xf16> -> tensor<?xf16>
                          %102 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %103 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %104 = loom.init_tensor %103[%22] : memref<?xf16> -> tensor<?xf16>
                          %105 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %106 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %107 = loom.semaphore_take %106 : memref<32x?xf16> -> memref<32x?xf16>
                          %108 = loom.init_tensor %107[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %109 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %110 = loom.semaphore_take %109 : memref<32x?xf16> -> memref<32x?xf16>
                          %111 = loom.init_tensor %110[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %112 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %116 = scf.for %arg18 = %c0 to %86 step %c1 iter_args(%arg19 = %83) -> (tensor<?x?xf16>) {
                            %131 = arith.muli %arg18, %22 : index
                            %132 = loom.subview %arg0[%41, %43, %60, %44, %131] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %132, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %133 = loom.bufferize_to_tensor %98[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %134 = loom.sync ins(%133 : tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            %135 = loom.subview %arg1[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %135, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %136 = loom.bufferize_to_tensor %105[%22] : memref<?xf16> -> tensor<?xf16>
                            %137 = loom.sync ins(%136 : tensor<?xf16>) outs(%104 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %105 : memref<?xf16>
                            %138 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%54 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %139 = loom.broadcast ins(%137 : tensor<?xf16>) outs(%108 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?xf16>
                            %140 = loom.subview %arg2[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %140, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %141 = loom.bufferize_to_tensor %102[%22] : memref<?xf16> -> tensor<?xf16>
                            %142 = loom.sync ins(%141 : tensor<?xf16>) outs(%101 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %102 : memref<?xf16>
                            %143 = loom.broadcast ins(%142 : tensor<?xf16>) outs(%111 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?xf16>
                            %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%134, %138, %139, %143 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %152 = arith.subf %in_0, %in_1 : f16
                              %153 = math.exp %152 : f16
                              %154 = arith.mulf %in, %153 : f16
                              %155 = arith.mulf %154, %in_2 : f16
                              linalg.yield %155 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<32x?xf16>
                            loom.semaphore_give %107 : memref<32x?xf16>
                            loom.semaphore_give %53 : memref<?x32xf16>
                            %145 = arith.addi %131, %58 : index
                            %146 = loom.subview %arg3[%41, %145, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %148 = loom.sync ins(%147 : tensor<?x?xf16>) outs(%114 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %115 : memref<?x?xf16>
                            %149 = linalg.fill ins(%cst : f16) outs(%94 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %150 = linalg.matmul ins(%144, %148 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%149 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %150 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %152 = arith.addf %in, %in_0 : f16
                              linalg.yield %152 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %151 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %117 = loom.alloc [1] on @L1 : memref<f16>
                          %118 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %119 = loom.init_tensor %118[] : memref<f16> -> tensor<f16>
                          %120 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %121 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %121, %120 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %122 = loom.bufferize_to_tensor %120[] : memref<f16> -> tensor<f16>
                          %123 = loom.sync ins(%122 : tensor<f16>) outs(%119 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %120 : memref<f16>
                          %124 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %124, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %125 = loom.bufferize_to_tensor %92[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %126 = loom.sync ins(%125 : tensor<?x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %92 : memref<?x?xf16>
                          %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %126, %123 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%91 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %131 = arith.mulf %in_0, %in_1 : f16
                            %132 = arith.addf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %118 : memref<f16>
                          loom.semaphore_give %79 : memref<?x?xf16>
                          %128 = loom.sync ins(%127 : tensor<?x?xf16>) outs(%89 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          %129 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %130 = loom.bufferize_to_memref %128 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %130, %129 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %88 : memref<?x?xf16>
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %50 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %51 = loom.sync ins(%50 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %52 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %53 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %54 = loom.init_tensor %53[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %55 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %56 = loom.init_tensor %55[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %57 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%56 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %58 = arith.muli %43, %c256 : index
                          %59 = arith.addi %44, %58 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.init_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %64 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %65 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %64[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = loom.sync ins(%66 : tensor<?x64xf16>) outs(%63 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %64 : memref<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %70 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %71 = loom.init_tensor %70[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %72 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %73 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %73, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %74 = loom.bufferize_to_tensor %72[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %75 = loom.sync ins(%74 : tensor<64x?xf16>) outs(%71 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %72 : memref<64x?xf16>
                          %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %77 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %79 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = linalg.fill ins(%cst : f16) outs(%78 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %82 = linalg.matmul ins(%67, %75 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%80 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %131 = math.exp %in_0 : f16
                            %132 = arith.mulf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<?x?xf16>
                          loom.semaphore_give %55 : memref<?x32xf16>
                          %84 = arith.addi %37, %c1 : index
                          %85 = arith.muli %84, %20 : index
                          %86 = arith.ceildivui %85, %22 : index
                          %87 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %89 = loom.init_tensor %88[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %90 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %91 = loom.init_tensor %90[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %92 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %93 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = loom.init_tensor %96[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %98 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %100 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %101 = loom.init_tensor %100[%22] : memref<?xf16> -> tensor<?xf16>
                          %102 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %103 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %104 = loom.init_tensor %103[%22] : memref<?xf16> -> tensor<?xf16>
                          %105 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %106 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %107 = loom.semaphore_take %106 : memref<32x?xf16> -> memref<32x?xf16>
                          %108 = loom.init_tensor %107[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %109 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %110 = loom.semaphore_take %109 : memref<32x?xf16> -> memref<32x?xf16>
                          %111 = loom.init_tensor %110[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %112 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %116 = scf.for %arg18 = %c0 to %86 step %c1 iter_args(%arg19 = %83) -> (tensor<?x?xf16>) {
                            %131 = arith.muli %arg18, %22 : index
                            %132 = loom.subview %arg0[%41, %43, %60, %44, %131] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %132, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %133 = loom.bufferize_to_tensor %98[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %134 = loom.sync ins(%133 : tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            %135 = loom.subview %arg1[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %135, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %136 = loom.bufferize_to_tensor %105[%22] : memref<?xf16> -> tensor<?xf16>
                            %137 = loom.sync ins(%136 : tensor<?xf16>) outs(%104 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %105 : memref<?xf16>
                            %138 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%54 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %139 = loom.broadcast ins(%137 : tensor<?xf16>) outs(%108 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?xf16>
                            %140 = loom.subview %arg2[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %140, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %141 = loom.bufferize_to_tensor %102[%22] : memref<?xf16> -> tensor<?xf16>
                            %142 = loom.sync ins(%141 : tensor<?xf16>) outs(%101 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %102 : memref<?xf16>
                            %143 = loom.broadcast ins(%142 : tensor<?xf16>) outs(%111 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?xf16>
                            %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%134, %138, %139, %143 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %152 = arith.subf %in_0, %in_1 : f16
                              %153 = math.exp %152 : f16
                              %154 = arith.mulf %in, %153 : f16
                              %155 = arith.mulf %154, %in_2 : f16
                              linalg.yield %155 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<32x?xf16>
                            loom.semaphore_give %107 : memref<32x?xf16>
                            loom.semaphore_give %53 : memref<?x32xf16>
                            %145 = arith.addi %131, %58 : index
                            %146 = loom.subview %arg3[%41, %145, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %148 = loom.sync ins(%147 : tensor<?x?xf16>) outs(%114 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %115 : memref<?x?xf16>
                            %149 = linalg.fill ins(%cst : f16) outs(%94 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %150 = linalg.matmul ins(%144, %148 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%149 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %150 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %152 = arith.addf %in, %in_0 : f16
                              linalg.yield %152 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %151 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %117 = loom.alloc [1] on @L1 : memref<f16>
                          %118 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %119 = loom.init_tensor %118[] : memref<f16> -> tensor<f16>
                          %120 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %121 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %121, %120 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %122 = loom.bufferize_to_tensor %120[] : memref<f16> -> tensor<f16>
                          %123 = loom.sync ins(%122 : tensor<f16>) outs(%119 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %120 : memref<f16>
                          %124 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %124, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %125 = loom.bufferize_to_tensor %92[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %126 = loom.sync ins(%125 : tensor<?x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %92 : memref<?x?xf16>
                          %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %126, %123 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%91 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %131 = arith.mulf %in_0, %in_1 : f16
                            %132 = arith.addf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %118 : memref<f16>
                          loom.semaphore_give %79 : memref<?x?xf16>
                          %128 = loom.sync ins(%127 : tensor<?x?xf16>) outs(%89 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          %129 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %130 = loom.bufferize_to_memref %128 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %130, %129 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %88 : memref<?x?xf16>
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %50 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %51 = loom.sync ins(%50 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %52 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %53 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %54 = loom.init_tensor %53[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %55 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %56 = loom.init_tensor %55[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %57 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%56 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %58 = arith.muli %43, %c256 : index
                          %59 = arith.addi %44, %58 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.init_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %64 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %65 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %64[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = loom.sync ins(%66 : tensor<?x64xf16>) outs(%63 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %64 : memref<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %70 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %71 = loom.init_tensor %70[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %72 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %73 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %73, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %74 = loom.bufferize_to_tensor %72[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %75 = loom.sync ins(%74 : tensor<64x?xf16>) outs(%71 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %72 : memref<64x?xf16>
                          %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %77 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %79 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = linalg.fill ins(%cst : f16) outs(%78 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %82 = linalg.matmul ins(%67, %75 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%80 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %131 = math.exp %in_0 : f16
                            %132 = arith.mulf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<?x?xf16>
                          loom.semaphore_give %55 : memref<?x32xf16>
                          %84 = arith.addi %37, %c1 : index
                          %85 = arith.muli %84, %20 : index
                          %86 = arith.ceildivui %85, %22 : index
                          %87 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %89 = loom.init_tensor %88[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %90 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %91 = loom.init_tensor %90[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %92 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %93 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = loom.init_tensor %96[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %98 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %100 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %101 = loom.init_tensor %100[%22] : memref<?xf16> -> tensor<?xf16>
                          %102 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %103 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %104 = loom.init_tensor %103[%22] : memref<?xf16> -> tensor<?xf16>
                          %105 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %106 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %107 = loom.semaphore_take %106 : memref<32x?xf16> -> memref<32x?xf16>
                          %108 = loom.init_tensor %107[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %109 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %110 = loom.semaphore_take %109 : memref<32x?xf16> -> memref<32x?xf16>
                          %111 = loom.init_tensor %110[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %112 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %116 = scf.for %arg18 = %c0 to %86 step %c1 iter_args(%arg19 = %83) -> (tensor<?x?xf16>) {
                            %131 = arith.muli %arg18, %22 : index
                            %132 = loom.subview %arg0[%41, %43, %60, %44, %131] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %132, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %133 = loom.bufferize_to_tensor %98[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %134 = loom.sync ins(%133 : tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            %135 = loom.subview %arg1[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %135, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %136 = loom.bufferize_to_tensor %105[%22] : memref<?xf16> -> tensor<?xf16>
                            %137 = loom.sync ins(%136 : tensor<?xf16>) outs(%104 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %105 : memref<?xf16>
                            %138 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%54 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %139 = loom.broadcast ins(%137 : tensor<?xf16>) outs(%108 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?xf16>
                            %140 = loom.subview %arg2[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %140, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %141 = loom.bufferize_to_tensor %102[%22] : memref<?xf16> -> tensor<?xf16>
                            %142 = loom.sync ins(%141 : tensor<?xf16>) outs(%101 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %102 : memref<?xf16>
                            %143 = loom.broadcast ins(%142 : tensor<?xf16>) outs(%111 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?xf16>
                            %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%134, %138, %139, %143 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %152 = arith.subf %in_0, %in_1 : f16
                              %153 = math.exp %152 : f16
                              %154 = arith.mulf %in, %153 : f16
                              %155 = arith.mulf %154, %in_2 : f16
                              linalg.yield %155 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<32x?xf16>
                            loom.semaphore_give %107 : memref<32x?xf16>
                            loom.semaphore_give %53 : memref<?x32xf16>
                            %145 = arith.addi %131, %58 : index
                            %146 = loom.subview %arg3[%41, %145, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %148 = loom.sync ins(%147 : tensor<?x?xf16>) outs(%114 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %115 : memref<?x?xf16>
                            %149 = linalg.fill ins(%cst : f16) outs(%94 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %150 = linalg.matmul ins(%144, %148 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%149 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %150 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %152 = arith.addf %in, %in_0 : f16
                              linalg.yield %152 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %151 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %117 = loom.alloc [1] on @L1 : memref<f16>
                          %118 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %119 = loom.init_tensor %118[] : memref<f16> -> tensor<f16>
                          %120 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %121 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %121, %120 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %122 = loom.bufferize_to_tensor %120[] : memref<f16> -> tensor<f16>
                          %123 = loom.sync ins(%122 : tensor<f16>) outs(%119 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %120 : memref<f16>
                          %124 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %124, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %125 = loom.bufferize_to_tensor %92[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %126 = loom.sync ins(%125 : tensor<?x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %92 : memref<?x?xf16>
                          %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %126, %123 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%91 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %131 = arith.mulf %in_0, %in_1 : f16
                            %132 = arith.addf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %118 : memref<f16>
                          loom.semaphore_give %79 : memref<?x?xf16>
                          %128 = loom.sync ins(%127 : tensor<?x?xf16>) outs(%89 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          %129 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %130 = loom.bufferize_to_memref %128 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %130, %129 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %88 : memref<?x?xf16>
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %50 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %51 = loom.sync ins(%50 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %52 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %53 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %54 = loom.init_tensor %53[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %55 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %56 = loom.init_tensor %55[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %57 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%56 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %58 = arith.muli %43, %c256 : index
                          %59 = arith.addi %44, %58 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.init_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %64 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %65 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %64[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = loom.sync ins(%66 : tensor<?x64xf16>) outs(%63 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %64 : memref<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %70 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %71 = loom.init_tensor %70[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %72 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %73 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %73, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %74 = loom.bufferize_to_tensor %72[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %75 = loom.sync ins(%74 : tensor<64x?xf16>) outs(%71 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %72 : memref<64x?xf16>
                          %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %77 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %79 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = linalg.fill ins(%cst : f16) outs(%78 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %82 = linalg.matmul ins(%67, %75 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%80 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %131 = math.exp %in_0 : f16
                            %132 = arith.mulf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<?x?xf16>
                          loom.semaphore_give %55 : memref<?x32xf16>
                          %84 = arith.addi %37, %c1 : index
                          %85 = arith.muli %84, %20 : index
                          %86 = arith.ceildivui %85, %22 : index
                          %87 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %89 = loom.init_tensor %88[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %90 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %91 = loom.init_tensor %90[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %92 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %93 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = loom.init_tensor %96[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %98 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %100 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %101 = loom.init_tensor %100[%22] : memref<?xf16> -> tensor<?xf16>
                          %102 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %103 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %104 = loom.init_tensor %103[%22] : memref<?xf16> -> tensor<?xf16>
                          %105 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %106 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %107 = loom.semaphore_take %106 : memref<32x?xf16> -> memref<32x?xf16>
                          %108 = loom.init_tensor %107[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %109 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %110 = loom.semaphore_take %109 : memref<32x?xf16> -> memref<32x?xf16>
                          %111 = loom.init_tensor %110[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %112 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %116 = scf.for %arg18 = %c0 to %86 step %c1 iter_args(%arg19 = %83) -> (tensor<?x?xf16>) {
                            %131 = arith.muli %arg18, %22 : index
                            %132 = loom.subview %arg0[%41, %43, %60, %44, %131] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %132, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %133 = loom.bufferize_to_tensor %98[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %134 = loom.sync ins(%133 : tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            %135 = loom.subview %arg1[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %135, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %136 = loom.bufferize_to_tensor %105[%22] : memref<?xf16> -> tensor<?xf16>
                            %137 = loom.sync ins(%136 : tensor<?xf16>) outs(%104 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %105 : memref<?xf16>
                            %138 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%54 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %139 = loom.broadcast ins(%137 : tensor<?xf16>) outs(%108 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?xf16>
                            %140 = loom.subview %arg2[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %140, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %141 = loom.bufferize_to_tensor %102[%22] : memref<?xf16> -> tensor<?xf16>
                            %142 = loom.sync ins(%141 : tensor<?xf16>) outs(%101 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %102 : memref<?xf16>
                            %143 = loom.broadcast ins(%142 : tensor<?xf16>) outs(%111 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?xf16>
                            %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%134, %138, %139, %143 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %152 = arith.subf %in_0, %in_1 : f16
                              %153 = math.exp %152 : f16
                              %154 = arith.mulf %in, %153 : f16
                              %155 = arith.mulf %154, %in_2 : f16
                              linalg.yield %155 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<32x?xf16>
                            loom.semaphore_give %107 : memref<32x?xf16>
                            loom.semaphore_give %53 : memref<?x32xf16>
                            %145 = arith.addi %131, %58 : index
                            %146 = loom.subview %arg3[%41, %145, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %148 = loom.sync ins(%147 : tensor<?x?xf16>) outs(%114 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %115 : memref<?x?xf16>
                            %149 = linalg.fill ins(%cst : f16) outs(%94 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %150 = linalg.matmul ins(%144, %148 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%149 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %150 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %152 = arith.addf %in, %in_0 : f16
                              linalg.yield %152 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %151 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %117 = loom.alloc [1] on @L1 : memref<f16>
                          %118 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %119 = loom.init_tensor %118[] : memref<f16> -> tensor<f16>
                          %120 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %121 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %121, %120 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %122 = loom.bufferize_to_tensor %120[] : memref<f16> -> tensor<f16>
                          %123 = loom.sync ins(%122 : tensor<f16>) outs(%119 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %120 : memref<f16>
                          %124 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %124, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %125 = loom.bufferize_to_tensor %92[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %126 = loom.sync ins(%125 : tensor<?x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %92 : memref<?x?xf16>
                          %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %126, %123 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%91 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %131 = arith.mulf %in_0, %in_1 : f16
                            %132 = arith.addf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %118 : memref<f16>
                          loom.semaphore_give %79 : memref<?x?xf16>
                          %128 = loom.sync ins(%127 : tensor<?x?xf16>) outs(%89 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          %129 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %130 = loom.bufferize_to_memref %128 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %130, %129 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %88 : memref<?x?xf16>
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %50 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %51 = loom.sync ins(%50 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %52 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %53 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %54 = loom.init_tensor %53[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %55 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %56 = loom.init_tensor %55[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %57 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%56 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %58 = arith.muli %43, %c256 : index
                          %59 = arith.addi %44, %58 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.init_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %64 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %65 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %64[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = loom.sync ins(%66 : tensor<?x64xf16>) outs(%63 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %64 : memref<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %70 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %71 = loom.init_tensor %70[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %72 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %73 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %73, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %74 = loom.bufferize_to_tensor %72[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %75 = loom.sync ins(%74 : tensor<64x?xf16>) outs(%71 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %72 : memref<64x?xf16>
                          %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %77 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %79 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = linalg.fill ins(%cst : f16) outs(%78 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %82 = linalg.matmul ins(%67, %75 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%80 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %131 = math.exp %in_0 : f16
                            %132 = arith.mulf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<?x?xf16>
                          loom.semaphore_give %55 : memref<?x32xf16>
                          %84 = arith.addi %37, %c1 : index
                          %85 = arith.muli %84, %20 : index
                          %86 = arith.ceildivui %85, %22 : index
                          %87 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %89 = loom.init_tensor %88[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %90 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %91 = loom.init_tensor %90[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %92 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %93 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = loom.init_tensor %96[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %98 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %100 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %101 = loom.init_tensor %100[%22] : memref<?xf16> -> tensor<?xf16>
                          %102 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %103 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %104 = loom.init_tensor %103[%22] : memref<?xf16> -> tensor<?xf16>
                          %105 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %106 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %107 = loom.semaphore_take %106 : memref<32x?xf16> -> memref<32x?xf16>
                          %108 = loom.init_tensor %107[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %109 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %110 = loom.semaphore_take %109 : memref<32x?xf16> -> memref<32x?xf16>
                          %111 = loom.init_tensor %110[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %112 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %116 = scf.for %arg18 = %c0 to %86 step %c1 iter_args(%arg19 = %83) -> (tensor<?x?xf16>) {
                            %131 = arith.muli %arg18, %22 : index
                            %132 = loom.subview %arg0[%41, %43, %60, %44, %131] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %132, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %133 = loom.bufferize_to_tensor %98[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %134 = loom.sync ins(%133 : tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            %135 = loom.subview %arg1[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %135, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %136 = loom.bufferize_to_tensor %105[%22] : memref<?xf16> -> tensor<?xf16>
                            %137 = loom.sync ins(%136 : tensor<?xf16>) outs(%104 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %105 : memref<?xf16>
                            %138 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%54 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %139 = loom.broadcast ins(%137 : tensor<?xf16>) outs(%108 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?xf16>
                            %140 = loom.subview %arg2[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %140, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %141 = loom.bufferize_to_tensor %102[%22] : memref<?xf16> -> tensor<?xf16>
                            %142 = loom.sync ins(%141 : tensor<?xf16>) outs(%101 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %102 : memref<?xf16>
                            %143 = loom.broadcast ins(%142 : tensor<?xf16>) outs(%111 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?xf16>
                            %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%134, %138, %139, %143 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %152 = arith.subf %in_0, %in_1 : f16
                              %153 = math.exp %152 : f16
                              %154 = arith.mulf %in, %153 : f16
                              %155 = arith.mulf %154, %in_2 : f16
                              linalg.yield %155 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<32x?xf16>
                            loom.semaphore_give %107 : memref<32x?xf16>
                            loom.semaphore_give %53 : memref<?x32xf16>
                            %145 = arith.addi %131, %58 : index
                            %146 = loom.subview %arg3[%41, %145, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %148 = loom.sync ins(%147 : tensor<?x?xf16>) outs(%114 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %115 : memref<?x?xf16>
                            %149 = linalg.fill ins(%cst : f16) outs(%94 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %150 = linalg.matmul ins(%144, %148 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%149 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %150 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %152 = arith.addf %in, %in_0 : f16
                              linalg.yield %152 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %151 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %117 = loom.alloc [1] on @L1 : memref<f16>
                          %118 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %119 = loom.init_tensor %118[] : memref<f16> -> tensor<f16>
                          %120 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %121 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %121, %120 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %122 = loom.bufferize_to_tensor %120[] : memref<f16> -> tensor<f16>
                          %123 = loom.sync ins(%122 : tensor<f16>) outs(%119 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %120 : memref<f16>
                          %124 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %124, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %125 = loom.bufferize_to_tensor %92[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %126 = loom.sync ins(%125 : tensor<?x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %92 : memref<?x?xf16>
                          %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %126, %123 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%91 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %131 = arith.mulf %in_0, %in_1 : f16
                            %132 = arith.addf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %118 : memref<f16>
                          loom.semaphore_give %79 : memref<?x?xf16>
                          %128 = loom.sync ins(%127 : tensor<?x?xf16>) outs(%89 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          %129 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %130 = loom.bufferize_to_memref %128 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %130, %129 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %88 : memref<?x?xf16>
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %50 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %51 = loom.sync ins(%50 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %52 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %53 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %54 = loom.init_tensor %53[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %55 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %56 = loom.init_tensor %55[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %57 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%56 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %58 = arith.muli %43, %c256 : index
                          %59 = arith.addi %44, %58 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.init_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %64 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %65 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %64[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = loom.sync ins(%66 : tensor<?x64xf16>) outs(%63 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %64 : memref<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %70 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %71 = loom.init_tensor %70[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %72 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %73 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %73, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %74 = loom.bufferize_to_tensor %72[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %75 = loom.sync ins(%74 : tensor<64x?xf16>) outs(%71 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %72 : memref<64x?xf16>
                          %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %77 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %79 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = linalg.fill ins(%cst : f16) outs(%78 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %82 = linalg.matmul ins(%67, %75 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%80 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %131 = math.exp %in_0 : f16
                            %132 = arith.mulf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<?x?xf16>
                          loom.semaphore_give %55 : memref<?x32xf16>
                          %84 = arith.addi %37, %c1 : index
                          %85 = arith.muli %84, %20 : index
                          %86 = arith.ceildivui %85, %22 : index
                          %87 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %89 = loom.init_tensor %88[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %90 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %91 = loom.init_tensor %90[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %92 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %93 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = loom.init_tensor %96[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %98 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %100 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %101 = loom.init_tensor %100[%22] : memref<?xf16> -> tensor<?xf16>
                          %102 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %103 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %104 = loom.init_tensor %103[%22] : memref<?xf16> -> tensor<?xf16>
                          %105 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %106 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %107 = loom.semaphore_take %106 : memref<32x?xf16> -> memref<32x?xf16>
                          %108 = loom.init_tensor %107[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %109 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %110 = loom.semaphore_take %109 : memref<32x?xf16> -> memref<32x?xf16>
                          %111 = loom.init_tensor %110[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %112 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %116 = scf.for %arg18 = %c0 to %86 step %c1 iter_args(%arg19 = %83) -> (tensor<?x?xf16>) {
                            %131 = arith.muli %arg18, %22 : index
                            %132 = loom.subview %arg0[%41, %43, %60, %44, %131] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %132, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %133 = loom.bufferize_to_tensor %98[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %134 = loom.sync ins(%133 : tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            %135 = loom.subview %arg1[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %135, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %136 = loom.bufferize_to_tensor %105[%22] : memref<?xf16> -> tensor<?xf16>
                            %137 = loom.sync ins(%136 : tensor<?xf16>) outs(%104 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %105 : memref<?xf16>
                            %138 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%54 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %139 = loom.broadcast ins(%137 : tensor<?xf16>) outs(%108 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?xf16>
                            %140 = loom.subview %arg2[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %140, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %141 = loom.bufferize_to_tensor %102[%22] : memref<?xf16> -> tensor<?xf16>
                            %142 = loom.sync ins(%141 : tensor<?xf16>) outs(%101 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %102 : memref<?xf16>
                            %143 = loom.broadcast ins(%142 : tensor<?xf16>) outs(%111 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?xf16>
                            %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%134, %138, %139, %143 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %152 = arith.subf %in_0, %in_1 : f16
                              %153 = math.exp %152 : f16
                              %154 = arith.mulf %in, %153 : f16
                              %155 = arith.mulf %154, %in_2 : f16
                              linalg.yield %155 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<32x?xf16>
                            loom.semaphore_give %107 : memref<32x?xf16>
                            loom.semaphore_give %53 : memref<?x32xf16>
                            %145 = arith.addi %131, %58 : index
                            %146 = loom.subview %arg3[%41, %145, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %148 = loom.sync ins(%147 : tensor<?x?xf16>) outs(%114 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %115 : memref<?x?xf16>
                            %149 = linalg.fill ins(%cst : f16) outs(%94 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %150 = linalg.matmul ins(%144, %148 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%149 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %150 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %152 = arith.addf %in, %in_0 : f16
                              linalg.yield %152 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %151 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %117 = loom.alloc [1] on @L1 : memref<f16>
                          %118 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %119 = loom.init_tensor %118[] : memref<f16> -> tensor<f16>
                          %120 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %121 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %121, %120 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %122 = loom.bufferize_to_tensor %120[] : memref<f16> -> tensor<f16>
                          %123 = loom.sync ins(%122 : tensor<f16>) outs(%119 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %120 : memref<f16>
                          %124 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %124, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %125 = loom.bufferize_to_tensor %92[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %126 = loom.sync ins(%125 : tensor<?x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %92 : memref<?x?xf16>
                          %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %126, %123 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%91 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %131 = arith.mulf %in_0, %in_1 : f16
                            %132 = arith.addf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %118 : memref<f16>
                          loom.semaphore_give %79 : memref<?x?xf16>
                          %128 = loom.sync ins(%127 : tensor<?x?xf16>) outs(%89 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          %129 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %130 = loom.bufferize_to_memref %128 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %130, %129 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %88 : memref<?x?xf16>
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %50 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %51 = loom.sync ins(%50 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %52 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %53 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %54 = loom.init_tensor %53[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %55 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %56 = loom.init_tensor %55[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %57 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%56 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %58 = arith.muli %43, %c256 : index
                          %59 = arith.addi %44, %58 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.init_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %64 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %65 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %64[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = loom.sync ins(%66 : tensor<?x64xf16>) outs(%63 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %64 : memref<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %70 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %71 = loom.init_tensor %70[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %72 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %73 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %73, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %74 = loom.bufferize_to_tensor %72[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %75 = loom.sync ins(%74 : tensor<64x?xf16>) outs(%71 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %72 : memref<64x?xf16>
                          %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %77 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %79 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = linalg.fill ins(%cst : f16) outs(%78 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %82 = linalg.matmul ins(%67, %75 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%80 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %131 = math.exp %in_0 : f16
                            %132 = arith.mulf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<?x?xf16>
                          loom.semaphore_give %55 : memref<?x32xf16>
                          %84 = arith.addi %37, %c1 : index
                          %85 = arith.muli %84, %20 : index
                          %86 = arith.ceildivui %85, %22 : index
                          %87 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %89 = loom.init_tensor %88[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %90 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %91 = loom.init_tensor %90[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %92 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %93 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = loom.init_tensor %96[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %98 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %100 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %101 = loom.init_tensor %100[%22] : memref<?xf16> -> tensor<?xf16>
                          %102 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %103 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %104 = loom.init_tensor %103[%22] : memref<?xf16> -> tensor<?xf16>
                          %105 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %106 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %107 = loom.semaphore_take %106 : memref<32x?xf16> -> memref<32x?xf16>
                          %108 = loom.init_tensor %107[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %109 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %110 = loom.semaphore_take %109 : memref<32x?xf16> -> memref<32x?xf16>
                          %111 = loom.init_tensor %110[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %112 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %116 = scf.for %arg18 = %c0 to %86 step %c1 iter_args(%arg19 = %83) -> (tensor<?x?xf16>) {
                            %131 = arith.muli %arg18, %22 : index
                            %132 = loom.subview %arg0[%41, %43, %60, %44, %131] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %132, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %133 = loom.bufferize_to_tensor %98[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %134 = loom.sync ins(%133 : tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            %135 = loom.subview %arg1[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %135, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %136 = loom.bufferize_to_tensor %105[%22] : memref<?xf16> -> tensor<?xf16>
                            %137 = loom.sync ins(%136 : tensor<?xf16>) outs(%104 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %105 : memref<?xf16>
                            %138 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%54 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %139 = loom.broadcast ins(%137 : tensor<?xf16>) outs(%108 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?xf16>
                            %140 = loom.subview %arg2[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %140, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %141 = loom.bufferize_to_tensor %102[%22] : memref<?xf16> -> tensor<?xf16>
                            %142 = loom.sync ins(%141 : tensor<?xf16>) outs(%101 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %102 : memref<?xf16>
                            %143 = loom.broadcast ins(%142 : tensor<?xf16>) outs(%111 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?xf16>
                            %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%134, %138, %139, %143 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %152 = arith.subf %in_0, %in_1 : f16
                              %153 = math.exp %152 : f16
                              %154 = arith.mulf %in, %153 : f16
                              %155 = arith.mulf %154, %in_2 : f16
                              linalg.yield %155 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<32x?xf16>
                            loom.semaphore_give %107 : memref<32x?xf16>
                            loom.semaphore_give %53 : memref<?x32xf16>
                            %145 = arith.addi %131, %58 : index
                            %146 = loom.subview %arg3[%41, %145, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %148 = loom.sync ins(%147 : tensor<?x?xf16>) outs(%114 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %115 : memref<?x?xf16>
                            %149 = linalg.fill ins(%cst : f16) outs(%94 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %150 = linalg.matmul ins(%144, %148 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%149 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %150 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %152 = arith.addf %in, %in_0 : f16
                              linalg.yield %152 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %151 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %117 = loom.alloc [1] on @L1 : memref<f16>
                          %118 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %119 = loom.init_tensor %118[] : memref<f16> -> tensor<f16>
                          %120 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %121 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %121, %120 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %122 = loom.bufferize_to_tensor %120[] : memref<f16> -> tensor<f16>
                          %123 = loom.sync ins(%122 : tensor<f16>) outs(%119 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %120 : memref<f16>
                          %124 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %124, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %125 = loom.bufferize_to_tensor %92[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %126 = loom.sync ins(%125 : tensor<?x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %92 : memref<?x?xf16>
                          %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %126, %123 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%91 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %131 = arith.mulf %in_0, %in_1 : f16
                            %132 = arith.addf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %118 : memref<f16>
                          loom.semaphore_give %79 : memref<?x?xf16>
                          %128 = loom.sync ins(%127 : tensor<?x?xf16>) outs(%89 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          %129 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %130 = loom.bufferize_to_memref %128 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %130, %129 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %88 : memref<?x?xf16>
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
                          %47 = loom.semaphore_take %45 : memref<?xf16> -> memref<?xf16>
                          %48 = loom.init_tensor %47[%20] : memref<?xf16> -> tensor<?xf16>
                          %49 = loom.subview %arg1[%41, %42, %43, %44] [1, 1, 1, %20] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                          loom.copy %49, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                          %50 = loom.bufferize_to_tensor %46[%20] : memref<?xf16> -> tensor<?xf16>
                          %51 = loom.sync ins(%50 : tensor<?xf16>) outs(%48 : tensor<?xf16>) -> tensor<?xf16>
                          loom.semaphore_give %46 : memref<?xf16>
                          %52 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
                          %53 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %54 = loom.init_tensor %53[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %55 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
                          %56 = loom.init_tensor %55[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
                          %57 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%56 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                          %58 = arith.muli %43, %c256 : index
                          %59 = arith.addi %44, %58 : index
                          %60 = arith.divui %42, %c64 : index
                          %61 = loom.alloc [%20, 64] on @L1 : memref<?x64xf16>
                          %62 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %63 = loom.init_tensor %62[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %64 = loom.semaphore_take %61 : memref<?x64xf16> -> memref<?x64xf16>
                          %65 = loom.subview %arg4[%41, %59, %60, 0] [1, %20, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
                          loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
                          %66 = loom.bufferize_to_tensor %64[%20, 64] : memref<?x64xf16> -> tensor<?x64xf16>
                          %67 = loom.sync ins(%66 : tensor<?x64xf16>) outs(%63 : tensor<?x64xf16>) -> tensor<?x64xf16>
                          loom.semaphore_give %64 : memref<?x64xf16>
                          %68 = arith.muli %38, %21 : index
                          %69 = loom.alloc [64, %21] on @L1 : memref<64x?xf16>
                          %70 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %71 = loom.init_tensor %70[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %72 = loom.semaphore_take %69 : memref<64x?xf16> -> memref<64x?xf16>
                          %73 = loom.subview %arg5[%41, %43, %42, 0, %68] [1, 1, 1, 64, %21] [1, 1, 1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
                          loom.copy %73, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
                          %74 = loom.bufferize_to_tensor %72[64, %21] : memref<64x?xf16> -> tensor<64x?xf16>
                          %75 = loom.sync ins(%74 : tensor<64x?xf16>) outs(%71 : tensor<64x?xf16>) -> tensor<64x?xf16>
                          loom.semaphore_give %72 : memref<64x?xf16>
                          %76 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %77 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %78 = loom.init_tensor %77[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %79 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
                          %80 = loom.init_tensor %79[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %81 = linalg.fill ins(%cst : f16) outs(%78 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          %82 = linalg.matmul ins(%67, %75 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %70 : memref<64x?xf16>
                          loom.semaphore_give %62 : memref<?x64xf16>
                          %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %57 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%80 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %out: f16):
                            %131 = math.exp %in_0 : f16
                            %132 = arith.mulf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %77 : memref<?x?xf16>
                          loom.semaphore_give %55 : memref<?x32xf16>
                          %84 = arith.addi %37, %c1 : index
                          %85 = arith.muli %84, %20 : index
                          %86 = arith.ceildivui %85, %22 : index
                          %87 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
                          %88 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %89 = loom.init_tensor %88[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %90 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %91 = loom.init_tensor %90[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %92 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %93 = loom.semaphore_take %87 : memref<?x?xf16> -> memref<?x?xf16>
                          %94 = loom.init_tensor %93[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %95 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
                          %96 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %97 = loom.init_tensor %96[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                          %98 = loom.semaphore_take %95 : memref<?x?xf16> -> memref<?x?xf16>
                          %99 = loom.alloc [%22] on @L1 : memref<?xf16>
                          %100 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %101 = loom.init_tensor %100[%22] : memref<?xf16> -> tensor<?xf16>
                          %102 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %103 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %104 = loom.init_tensor %103[%22] : memref<?xf16> -> tensor<?xf16>
                          %105 = loom.semaphore_take %99 : memref<?xf16> -> memref<?xf16>
                          %106 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %107 = loom.semaphore_take %106 : memref<32x?xf16> -> memref<32x?xf16>
                          %108 = loom.init_tensor %107[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %109 = loom.alloc [32, %22] on @L1 : memref<32x?xf16>
                          %110 = loom.semaphore_take %109 : memref<32x?xf16> -> memref<32x?xf16>
                          %111 = loom.init_tensor %110[32, %22] : memref<32x?xf16> -> tensor<32x?xf16>
                          %112 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
                          %113 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %114 = loom.init_tensor %113[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %115 = loom.semaphore_take %112 : memref<?x?xf16> -> memref<?x?xf16>
                          %116 = scf.for %arg18 = %c0 to %86 step %c1 iter_args(%arg19 = %83) -> (tensor<?x?xf16>) {
                            %131 = arith.muli %arg18, %22 : index
                            %132 = loom.subview %arg0[%41, %43, %60, %44, %131] [1, 1, 1, %20, %22] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                            loom.copy %132, %98 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                            %133 = loom.bufferize_to_tensor %98[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                            %134 = loom.sync ins(%133 : tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %98 : memref<?x?xf16>
                            %135 = loom.subview %arg1[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %135, %105 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %136 = loom.bufferize_to_tensor %105[%22] : memref<?xf16> -> tensor<?xf16>
                            %137 = loom.sync ins(%136 : tensor<?xf16>) outs(%104 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %105 : memref<?xf16>
                            %138 = loom.broadcast ins(%51 : tensor<?xf16>) outs(%54 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
                            %139 = loom.broadcast ins(%137 : tensor<?xf16>) outs(%108 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %103 : memref<?xf16>
                            %140 = loom.subview %arg2[%41, %42, %43, %131] [1, 1, 1, %22] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                            loom.copy %140, %102 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
                            %141 = loom.bufferize_to_tensor %102[%22] : memref<?xf16> -> tensor<?xf16>
                            %142 = loom.sync ins(%141 : tensor<?xf16>) outs(%101 : tensor<?xf16>) -> tensor<?xf16>
                            loom.semaphore_give %102 : memref<?xf16>
                            %143 = loom.broadcast ins(%142 : tensor<?xf16>) outs(%111 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
                            loom.semaphore_give %100 : memref<?xf16>
                            %144 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%134, %138, %139, %143 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
                              %152 = arith.subf %in_0, %in_1 : f16
                              %153 = math.exp %152 : f16
                              %154 = arith.mulf %in, %153 : f16
                              %155 = arith.mulf %154, %in_2 : f16
                              linalg.yield %155 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %110 : memref<32x?xf16>
                            loom.semaphore_give %107 : memref<32x?xf16>
                            loom.semaphore_give %53 : memref<?x32xf16>
                            %145 = arith.addi %131, %58 : index
                            %146 = loom.subview %arg3[%41, %145, %42, %68] [1, %22, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = true] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                            loom.copy %146, %115 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                            %147 = loom.bufferize_to_tensor %115[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                            %148 = loom.sync ins(%147 : tensor<?x?xf16>) outs(%114 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %115 : memref<?x?xf16>
                            %149 = linalg.fill ins(%cst : f16) outs(%94 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            %150 = linalg.matmul ins(%144, %148 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%149 : tensor<?x?xf16>) -> tensor<?x?xf16>
                            loom.semaphore_give %113 : memref<?x?xf16>
                            loom.semaphore_give %96 : memref<?x?xf16>
                            %151 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg19, %150 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg19 : tensor<?x?xf16>) {
                            ^bb0(%in: f16, %in_0: f16, %out: f16):
                              %152 = arith.addf %in, %in_0 : f16
                              linalg.yield %152 : f16
                            } -> tensor<?x?xf16>
                            loom.semaphore_give %93 : memref<?x?xf16>
                            scf.yield %151 : tensor<?x?xf16>
                          } {loom.iter_type = #loom.iter_type<sequential>}
                          loom.semaphore_give %47 : memref<?xf16>
                          %117 = loom.alloc [1] on @L1 : memref<f16>
                          %118 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %119 = loom.init_tensor %118[] : memref<f16> -> tensor<f16>
                          %120 = loom.semaphore_take %117 : memref<f16> -> memref<f16>
                          %121 = loom.subview %arg6[%42] [1] [1], reuse : [seq = false, spat = true, temp = true] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
                          loom.copy %121, %120 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
                          %122 = loom.bufferize_to_tensor %120[] : memref<f16> -> tensor<f16>
                          %123 = loom.sync ins(%122 : tensor<f16>) outs(%119 : tensor<f16>) -> tensor<f16>
                          loom.semaphore_give %120 : memref<f16>
                          %124 = loom.subview %arg3[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.copy %124, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                          %125 = loom.bufferize_to_tensor %92[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                          %126 = loom.sync ins(%125 : tensor<?x?xf16>) outs(%91 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %92 : memref<?x?xf16>
                          %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%116, %126, %123 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%91 : tensor<?x?xf16>) {
                          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
                            %131 = arith.mulf %in_0, %in_1 : f16
                            %132 = arith.addf %in, %131 : f16
                            linalg.yield %132 : f16
                          } -> tensor<?x?xf16>
                          loom.semaphore_give %118 : memref<f16>
                          loom.semaphore_give %79 : memref<?x?xf16>
                          %128 = loom.sync ins(%127 : tensor<?x?xf16>) outs(%89 : tensor<?x?xf16>) -> tensor<?x?xf16>
                          loom.semaphore_give %90 : memref<?x?xf16>
                          %129 = loom.subview %arg7[%41, %59, %42, %68] [1, %20, 1, %21] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          %130 = loom.bufferize_to_memref %128 : tensor<?x?xf16> -> memref<?x?xf16>
                          loom.copy %130, %129 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                          loom.semaphore_give %88 : memref<?x?xf16>
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
