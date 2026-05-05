module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y1y8__d0i1_d1i1_d2i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (1) {
            %25 = arith.ceildivui %23, %c8 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c8 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %28, %21 : index
                %35 = arith.ceildivui %21, %22 : index
                %36 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %110 = arith.muli %arg9, %22 : index
                  %111 = arith.addi %34, %110 : index
                  %112 = loom.subview %arg0[%29, 0, %111] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %112, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %113 = loom.bufferize_to_tensor %65[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %115 = linalg.batch_matmul ins(%33, %113 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%114 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %116 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%115 : tensor<?x32x?xf16>) outs(%116 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.maximumf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in_3, %cst_2 : f16
                    %133 = arith.cmpf ogt, %in, %132 : f16
                    %134 = arith.select %133, %in, %132 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = loom.broadcast ins(%118 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %119 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in, %cst_2 : f16
                    %133 = arith.subf %132, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %121 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%120 : tensor<?x32x?xf16>) outs(%121 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.addf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.subf %in, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %123, %122 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in, %in_3 : f16
                    %133 = arith.addf %132, %in_4 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %125 = loom.broadcast ins(%123 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %126 = loom.subview %arg1[%29, %111, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %126, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %127 = loom.bufferize_to_tensor %77[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %129 = linalg.batch_matmul ins(%120, %127 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%128 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %arg12, %125 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in_3, %in_4 : f16
                    %133 = arith.addf %in, %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %131 = linalg.copy ins(%118 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %131, %124, %130 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %79 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = math.log %in : f16
                  %111 = arith.addf %110, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = arith.divf %in, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %90 = loom.init_tensor %89[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %91 = arith.addi %arg6, %arg4 : index
                %92 = loom.gather ins(%82 : tensor<?x32x1xf16>) outs(%90 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %91], LR : [%c7, %91]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %93 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %95 = loom.init_tensor %94[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %96 = loom.gather ins(%87 : tensor<?x32x128xf16>) outs(%95 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %91], LR : [%c7, %91]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %97 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %98 = loom.semaphore_take %97 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %99 = loom.init_tensor %98[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %100 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %101 = loom.semaphore_take %100 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %102 = loom.init_tensor %101[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %103 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %104 = loom.semaphore_take %103 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %105 = loom.init_tensor %104[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %106 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %107 = loom.semaphore_take %106 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %108 = loom.init_tensor %107[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %109 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %109 {
                  %110 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%92 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.maximumf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%92, %111 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.subf %in, %in_3 : f16
                    %123 = math.exp %122 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %89 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%112 : tensor<?x?x32x1xf16>) outs(%113 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %114 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.divf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %116 = loom.broadcast ins(%115 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %101 : memref<?x?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%96, %116 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%105 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.mulf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x32xf16>
                  loom.semaphore_give %94 : memref<?x?x32x128xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%99 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%117 : tensor<?x?x32x128xf16>) outs(%118 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x128xf16>
                  %120 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %121 = loom.bufferize_to_memref %119 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %121, %120 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %98 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y2y4__d0i1_d1i1_d2i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (2) {
            %25 = arith.ceildivui %23, %c4 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c16 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg5, %arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %28, %21 : index
                %35 = arith.ceildivui %21, %22 : index
                %36 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %111 = arith.muli %arg9, %22 : index
                  %112 = arith.addi %34, %111 : index
                  %113 = loom.subview %arg0[%29, 0, %112] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %113, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %114 = loom.bufferize_to_tensor %65[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %116 = linalg.batch_matmul ins(%33, %114 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%115 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %117 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in_3, %cst_2 : f16
                    %134 = arith.cmpf ogt, %in, %133 : f16
                    %135 = arith.select %134, %in, %133 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %120 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in, %cst_2 : f16
                    %134 = arith.subf %133, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %122 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121 : tensor<?x32x?xf16>) outs(%122 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.subf %in, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %124, %123 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in, %in_3 : f16
                    %134 = arith.addf %133, %in_4 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %126 = loom.broadcast ins(%124 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %127 = loom.subview %arg1[%29, %112, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %127, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %128 = loom.bufferize_to_tensor %77[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %129 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %130 = linalg.batch_matmul ins(%121, %128 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%129 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%130, %arg12, %126 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in_3, %in_4 : f16
                    %134 = arith.addf %in, %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %132 = linalg.copy ins(%119 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %132, %125, %131 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %79 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = math.log %in : f16
                  %112 = arith.addf %111, %in_3 : f16
                  linalg.yield %112 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = arith.divf %in, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %90 = loom.init_tensor %89[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %91 = arith.muli %arg4, %c2 : index
                %92 = arith.addi %arg6, %91 : index
                %93 = loom.gather ins(%82 : tensor<?x32x1xf16>) outs(%90 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %92], LR : [%c7, %92]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %94 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %95 = loom.semaphore_take %94 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %96 = loom.init_tensor %95[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %97 = loom.gather ins(%87 : tensor<?x32x128xf16>) outs(%96 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %92], LR : [%c7, %92]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %98 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %99 = loom.semaphore_take %98 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %100 = loom.init_tensor %99[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %101 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %102 = loom.semaphore_take %101 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %103 = loom.init_tensor %102[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %104 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %106 = loom.init_tensor %105[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %107 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %108 = loom.semaphore_take %107 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %109 = loom.init_tensor %108[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %110 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %110 {
                  %111 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%93 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.maximumf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %112 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%103 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.subf %in, %in_3 : f16
                    %124 = math.exp %123 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %89 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%113 : tensor<?x?x32x1xf16>) outs(%114 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%113, %115 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%103 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.divf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %117 = loom.broadcast ins(%116 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %102 : memref<?x?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %117 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%106 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.mulf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %108 : memref<?x?x32x32xf16>
                  loom.semaphore_give %95 : memref<?x?x32x128xf16>
                  %119 = linalg.fill ins(%cst : f16) outs(%100 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%118 : tensor<?x?x32x128xf16>) outs(%119 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x128xf16>
                  %121 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %122 = loom.bufferize_to_memref %120 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %122, %121 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %99 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y4y2__d0i1_d1i1_d2i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (4) {
            %25 = arith.ceildivui %23, %c2 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c32 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg5, %arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %28, %21 : index
                %35 = arith.ceildivui %21, %22 : index
                %36 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %111 = arith.muli %arg9, %22 : index
                  %112 = arith.addi %34, %111 : index
                  %113 = loom.subview %arg0[%29, 0, %112] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %113, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %114 = loom.bufferize_to_tensor %65[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %116 = linalg.batch_matmul ins(%33, %114 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%115 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %117 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in_3, %cst_2 : f16
                    %134 = arith.cmpf ogt, %in, %133 : f16
                    %135 = arith.select %134, %in, %133 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %120 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in, %cst_2 : f16
                    %134 = arith.subf %133, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %122 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121 : tensor<?x32x?xf16>) outs(%122 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.subf %in, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %124, %123 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in, %in_3 : f16
                    %134 = arith.addf %133, %in_4 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %126 = loom.broadcast ins(%124 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %127 = loom.subview %arg1[%29, %112, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %127, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %128 = loom.bufferize_to_tensor %77[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %129 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %130 = linalg.batch_matmul ins(%121, %128 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%129 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%130, %arg12, %126 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in_3, %in_4 : f16
                    %134 = arith.addf %in, %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %132 = linalg.copy ins(%119 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %132, %125, %131 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %79 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = math.log %in : f16
                  %112 = arith.addf %111, %in_3 : f16
                  linalg.yield %112 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = arith.divf %in, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %90 = loom.init_tensor %89[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %91 = arith.muli %arg4, %c4 : index
                %92 = arith.addi %arg6, %91 : index
                %93 = loom.gather ins(%82 : tensor<?x32x1xf16>) outs(%90 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %92], LR : [%c7, %92]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %94 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %95 = loom.semaphore_take %94 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %96 = loom.init_tensor %95[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %97 = loom.gather ins(%87 : tensor<?x32x128xf16>) outs(%96 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %92], LR : [%c7, %92]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %98 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %99 = loom.semaphore_take %98 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %100 = loom.init_tensor %99[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %101 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %102 = loom.semaphore_take %101 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %103 = loom.init_tensor %102[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %104 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %106 = loom.init_tensor %105[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %107 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %108 = loom.semaphore_take %107 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %109 = loom.init_tensor %108[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %110 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %110 {
                  %111 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%93 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.maximumf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %112 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%103 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.subf %in, %in_3 : f16
                    %124 = math.exp %123 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %89 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%113 : tensor<?x?x32x1xf16>) outs(%114 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%113, %115 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%103 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.divf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %117 = loom.broadcast ins(%116 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %102 : memref<?x?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %117 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%106 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.mulf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %108 : memref<?x?x32x32xf16>
                  loom.semaphore_give %95 : memref<?x?x32x128xf16>
                  %119 = linalg.fill ins(%cst : f16) outs(%100 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%118 : tensor<?x?x32x128xf16>) outs(%119 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x128xf16>
                  %121 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %122 = loom.bufferize_to_memref %120 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %122, %121 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %99 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y8y1__d0i1_d1i1_d2i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %23 step %c1 {
              %25 = arith.ceildivui %24, %c64 : index
              scf.for %arg8 = %c0 to %25 step %c1 {
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %27 = arith.muli %arg7, %20 : index
                %28 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %30 = loom.subview %arg3[%27, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %31 = loom.bufferize_to_tensor %29[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %32 = arith.muli %26, %21 : index
                %33 = arith.ceildivui %21, %22 : index
                %34 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %36 = loom.init_tensor %35[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %38 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %39 = loom.semaphore_take %38 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %40 = loom.init_tensor %39[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %41 = loom.semaphore_take %38 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %38 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = linalg.fill ins(%cst_0 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %46 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %47 = loom.semaphore_take %46 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %48 = loom.init_tensor %47[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %49 = linalg.fill ins(%cst_1 : f16) outs(%48 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %50 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %51 = loom.semaphore_take %50 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %52 = loom.init_tensor %51[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %53 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %54 = loom.semaphore_take %53 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %55 = loom.init_tensor %54[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %56 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %57 = loom.semaphore_take %56 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %58 = loom.init_tensor %57[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %59 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %61 = loom.init_tensor %60[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %64 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %66 = loom.init_tensor %65[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %67 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %68 = loom.semaphore_take %67 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %69 = loom.init_tensor %68[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %70 = loom.semaphore_take %67 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %67 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %76:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %49, %arg11 = %45, %arg12 = %37) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %109 = arith.muli %arg9, %22 : index
                  %110 = arith.addi %32, %109 : index
                  %111 = loom.subview %arg0[%27, 0, %110] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %111, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %112 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%66 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %114 = linalg.batch_matmul ins(%31, %112 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%113 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %63 : memref<?x128x?xf16>
                  loom.semaphore_give %29 : memref<?x32x128xf16>
                  %115 = linalg.fill ins(%cst_1 : f16) outs(%55 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%114 : tensor<?x32x?xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.maximumf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %116 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%55 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in_3, %cst_2 : f16
                    %132 = arith.cmpf ogt, %in, %131 : f16
                    %133 = arith.select %132, %in, %131 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%114, %118 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%66 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in, %cst_2 : f16
                    %132 = arith.subf %131, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%58 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.addf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%61 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.subf %in, %in_3 : f16
                    %132 = math.exp %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %122, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in, %in_3 : f16
                    %132 = arith.addf %131, %in_4 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %57 : memref<?x32x1xf16>
                  %124 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %60 : memref<?x32x1xf16>
                  %125 = loom.subview %arg1[%27, %110, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %125, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %126 = loom.bufferize_to_tensor %75[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %127 = linalg.fill ins(%cst : f16) outs(%52 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %128 = linalg.batch_matmul ins(%119, %126 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%127 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %75 : memref<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x32x?xf16>
                  %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %arg12, %124 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in_3, %in_4 : f16
                    %132 = arith.addf %in, %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %70 : memref<?x32x32xf16>
                  loom.semaphore_give %51 : memref<?x32x128xf16>
                  %130 = linalg.copy ins(%117 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %54 : memref<?x32x1xf16>
                  scf.yield %130, %123, %129 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %77 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %78 = loom.semaphore_take %77 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %79 = loom.init_tensor %78[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76#1, %76#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%79 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = math.log %in : f16
                  %110 = arith.addf %109, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %47 : memref<?x32x1xf16>
                %81 = loom.broadcast ins(%76#1 : tensor<?x32x1xf16>) outs(%69 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %43 : memref<?x32x1xf16>
                %82 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %83 = loom.semaphore_take %82 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %84 = loom.init_tensor %83[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76#2, %81 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%84 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = arith.divf %in, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %68 : memref<?x32x32xf16>
                loom.semaphore_give %35 : memref<?x32x128xf16>
                %86 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %87 = loom.semaphore_take %86 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %88 = loom.init_tensor %87[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %89 = arith.muli %arg4, %c8 : index
                %90 = arith.addi %arg6, %89 : index
                %91 = loom.gather ins(%80 : tensor<?x32x1xf16>) outs(%88 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %90], LR : [%c7, %90]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %78 : memref<?x32x1xf16>
                %92 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %93 = loom.semaphore_take %92 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %94 = loom.init_tensor %93[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %95 = loom.gather ins(%85 : tensor<?x32x128xf16>) outs(%94 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %90], LR : [%c7, %90]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %83 : memref<?x32x128xf16>
                %96 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %98 = loom.init_tensor %97[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %99 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %100 = loom.semaphore_take %99 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %101 = loom.init_tensor %100[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %102 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %104 = loom.init_tensor %103[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %105 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %107 = loom.init_tensor %106[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %108 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %108 {
                  %109 = linalg.fill ins(%cst_1 : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%91 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.maximumf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.subf %in, %in_3 : f16
                    %122 = math.exp %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %87 : memref<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%40 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.divf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %39 : memref<?x32x1xf16>
                  %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%107 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %100 : memref<?x?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%104 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.mulf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x32xf16>
                  loom.semaphore_give %93 : memref<?x?x32x128xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%98 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%116 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x128xf16>
                  %119 = loom.subview %arg2[%27, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %120, %119 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %97 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x1x8_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (1) {
          affine.parallel (%arg6) = (0) to (8) {
            %25 = arith.ceildivui %23, %c8 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c8 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %28, %21 : index
                %35 = arith.ceildivui %21, %22 : index
                %36 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %109 = arith.muli %arg9, %22 : index
                  %110 = arith.addi %34, %109 : index
                  %111 = loom.subview %arg0[%29, 0, %110] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %111, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %112 = loom.bufferize_to_tensor %65[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %114 = linalg.batch_matmul ins(%33, %112 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%113 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %115 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%114 : tensor<?x32x?xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.maximumf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %116 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in_3, %cst_2 : f16
                    %132 = arith.cmpf ogt, %in, %131 : f16
                    %133 = arith.select %132, %in, %131 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%114, %118 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in, %cst_2 : f16
                    %132 = arith.subf %131, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.addf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.subf %in, %in_3 : f16
                    %132 = math.exp %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %122, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in, %in_3 : f16
                    %132 = arith.addf %131, %in_4 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %124 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %125 = loom.subview %arg1[%29, %110, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %125, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %126 = loom.bufferize_to_tensor %77[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %127 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %128 = linalg.batch_matmul ins(%119, %126 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%127 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %arg12, %124 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in_3, %in_4 : f16
                    %132 = arith.addf %in, %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %130 = linalg.copy ins(%117 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %130, %123, %129 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %79 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = math.log %in : f16
                  %110 = arith.addf %109, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = arith.divf %in, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %90 = loom.init_tensor %89[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %91 = loom.gather ins(%82 : tensor<?x32x1xf16>) outs(%90 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%arg4, %arg6], LR : [%arg4, %arg6]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %92 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %93 = loom.semaphore_take %92 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %94 = loom.init_tensor %93[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %95 = loom.gather ins(%87 : tensor<?x32x128xf16>) outs(%94 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%arg4, %arg6], LR : [%arg4, %arg6]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %96 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %98 = loom.init_tensor %97[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %99 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %100 = loom.semaphore_take %99 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %101 = loom.init_tensor %100[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %102 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %104 = loom.init_tensor %103[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %105 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %107 = loom.init_tensor %106[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %108 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %108 {
                  %109 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%91 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.maximumf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.subf %in, %in_3 : f16
                    %122 = math.exp %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %89 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.divf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%107 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %100 : memref<?x?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%104 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.mulf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x32xf16>
                  loom.semaphore_give %93 : memref<?x?x32x128xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%98 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%116 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x128xf16>
                  %119 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %120, %119 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %97 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x2x4_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (2) {
          affine.parallel (%arg6) = (0) to (8) {
            %25 = arith.ceildivui %23, %c4 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c16 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg5, %arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %28, %21 : index
                %35 = arith.ceildivui %21, %22 : index
                %36 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %111 = arith.muli %arg9, %22 : index
                  %112 = arith.addi %34, %111 : index
                  %113 = loom.subview %arg0[%29, 0, %112] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %113, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %114 = loom.bufferize_to_tensor %65[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %116 = linalg.batch_matmul ins(%33, %114 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%115 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %117 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in_3, %cst_2 : f16
                    %134 = arith.cmpf ogt, %in, %133 : f16
                    %135 = arith.select %134, %in, %133 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %120 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in, %cst_2 : f16
                    %134 = arith.subf %133, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %122 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121 : tensor<?x32x?xf16>) outs(%122 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.subf %in, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %124, %123 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in, %in_3 : f16
                    %134 = arith.addf %133, %in_4 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %126 = loom.broadcast ins(%124 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %127 = loom.subview %arg1[%29, %112, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %127, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %128 = loom.bufferize_to_tensor %77[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %129 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %130 = linalg.batch_matmul ins(%121, %128 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%129 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%130, %arg12, %126 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in_3, %in_4 : f16
                    %134 = arith.addf %in, %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %132 = linalg.copy ins(%119 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %132, %125, %131 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %79 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = math.log %in : f16
                  %112 = arith.addf %111, %in_3 : f16
                  linalg.yield %112 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = arith.divf %in, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %90 = loom.init_tensor %89[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %91 = arith.muli %arg4, %c2 : index
                %92 = arith.addi %91, %c1 : index
                %93 = loom.gather ins(%82 : tensor<?x32x1xf16>) outs(%90 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%91, %arg6], LR : [%92, %arg6]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %94 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %95 = loom.semaphore_take %94 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %96 = loom.init_tensor %95[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %97 = loom.gather ins(%87 : tensor<?x32x128xf16>) outs(%96 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%91, %arg6], LR : [%92, %arg6]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %98 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %99 = loom.semaphore_take %98 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %100 = loom.init_tensor %99[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %101 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %102 = loom.semaphore_take %101 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %103 = loom.init_tensor %102[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %104 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %106 = loom.init_tensor %105[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %107 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %108 = loom.semaphore_take %107 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %109 = loom.init_tensor %108[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %110 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %110 {
                  %111 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%93 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.maximumf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %112 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%103 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.subf %in, %in_3 : f16
                    %124 = math.exp %123 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %89 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%113 : tensor<?x?x32x1xf16>) outs(%114 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%113, %115 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%103 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.divf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %117 = loom.broadcast ins(%116 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %102 : memref<?x?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %117 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%106 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.mulf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %108 : memref<?x?x32x32xf16>
                  loom.semaphore_give %95 : memref<?x?x32x128xf16>
                  %119 = linalg.fill ins(%cst : f16) outs(%100 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%118 : tensor<?x?x32x128xf16>) outs(%119 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x128xf16>
                  %121 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %122 = loom.bufferize_to_memref %120 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %122, %121 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %99 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x4x2_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (4) {
          affine.parallel (%arg6) = (0) to (8) {
            %25 = arith.ceildivui %23, %c2 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c32 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg5, %arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %28, %21 : index
                %35 = arith.ceildivui %21, %22 : index
                %36 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %111 = arith.muli %arg9, %22 : index
                  %112 = arith.addi %34, %111 : index
                  %113 = loom.subview %arg0[%29, 0, %112] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %113, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %114 = loom.bufferize_to_tensor %65[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %116 = linalg.batch_matmul ins(%33, %114 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%115 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %117 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in_3, %cst_2 : f16
                    %134 = arith.cmpf ogt, %in, %133 : f16
                    %135 = arith.select %134, %in, %133 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %120 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in, %cst_2 : f16
                    %134 = arith.subf %133, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %122 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121 : tensor<?x32x?xf16>) outs(%122 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.subf %in, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %124, %123 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in, %in_3 : f16
                    %134 = arith.addf %133, %in_4 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %126 = loom.broadcast ins(%124 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %127 = loom.subview %arg1[%29, %112, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %127, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %128 = loom.bufferize_to_tensor %77[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %129 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %130 = linalg.batch_matmul ins(%121, %128 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%129 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%130, %arg12, %126 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in_3, %in_4 : f16
                    %134 = arith.addf %in, %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %132 = linalg.copy ins(%119 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %132, %125, %131 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %79 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = math.log %in : f16
                  %112 = arith.addf %111, %in_3 : f16
                  linalg.yield %112 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = arith.divf %in, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %90 = loom.init_tensor %89[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %91 = arith.muli %arg4, %c4 : index
                %92 = arith.addi %91, %c3 : index
                %93 = loom.gather ins(%82 : tensor<?x32x1xf16>) outs(%90 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%91, %arg6], LR : [%92, %arg6]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %94 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %95 = loom.semaphore_take %94 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %96 = loom.init_tensor %95[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %97 = loom.gather ins(%87 : tensor<?x32x128xf16>) outs(%96 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%91, %arg6], LR : [%92, %arg6]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %98 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %99 = loom.semaphore_take %98 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %100 = loom.init_tensor %99[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %101 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %102 = loom.semaphore_take %101 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %103 = loom.init_tensor %102[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %104 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %106 = loom.init_tensor %105[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %107 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %108 = loom.semaphore_take %107 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %109 = loom.init_tensor %108[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %110 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %110 {
                  %111 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%93 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.maximumf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %112 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%103 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.subf %in, %in_3 : f16
                    %124 = math.exp %123 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %89 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%113 : tensor<?x?x32x1xf16>) outs(%114 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%113, %115 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%103 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.divf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %117 = loom.broadcast ins(%116 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %102 : memref<?x?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %117 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%106 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.mulf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %108 : memref<?x?x32x32xf16>
                  loom.semaphore_give %95 : memref<?x?x32x128xf16>
                  %119 = linalg.fill ins(%cst : f16) outs(%100 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%118 : tensor<?x?x32x128xf16>) outs(%119 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x128xf16>
                  %121 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %122 = loom.bufferize_to_memref %120 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %122, %121 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %99 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8x1_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %23 step %c1 {
              %25 = arith.ceildivui %24, %c64 : index
              scf.for %arg8 = %c0 to %25 step %c1 {
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %27 = arith.muli %arg7, %20 : index
                %28 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %30 = loom.subview %arg3[%27, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %31 = loom.bufferize_to_tensor %29[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %32 = arith.muli %26, %21 : index
                %33 = arith.ceildivui %21, %22 : index
                %34 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %36 = loom.init_tensor %35[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %38 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %39 = loom.semaphore_take %38 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %40 = loom.init_tensor %39[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %41 = loom.semaphore_take %38 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %38 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = linalg.fill ins(%cst_0 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %46 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %47 = loom.semaphore_take %46 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %48 = loom.init_tensor %47[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %49 = linalg.fill ins(%cst_1 : f16) outs(%48 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %50 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %51 = loom.semaphore_take %50 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %52 = loom.init_tensor %51[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %53 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %54 = loom.semaphore_take %53 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %55 = loom.init_tensor %54[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %56 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %57 = loom.semaphore_take %56 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %58 = loom.init_tensor %57[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %59 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %61 = loom.init_tensor %60[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %62 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %64 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %66 = loom.init_tensor %65[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %67 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %68 = loom.semaphore_take %67 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %69 = loom.init_tensor %68[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %70 = loom.semaphore_take %67 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %67 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %76:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %49, %arg11 = %45, %arg12 = %37) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %109 = arith.muli %arg9, %22 : index
                  %110 = arith.addi %32, %109 : index
                  %111 = loom.subview %arg0[%27, 0, %110] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %111, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %112 = loom.bufferize_to_tensor %63[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%66 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %114 = linalg.batch_matmul ins(%31, %112 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%113 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %63 : memref<?x128x?xf16>
                  loom.semaphore_give %29 : memref<?x32x128xf16>
                  %115 = linalg.fill ins(%cst_1 : f16) outs(%55 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%114 : tensor<?x32x?xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.maximumf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %116 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%55 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in_3, %cst_2 : f16
                    %132 = arith.cmpf ogt, %in, %131 : f16
                    %133 = arith.select %132, %in, %131 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%114, %118 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%66 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in, %cst_2 : f16
                    %132 = arith.subf %131, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%58 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.addf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%61 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.subf %in, %in_3 : f16
                    %132 = math.exp %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %122, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in, %in_3 : f16
                    %132 = arith.addf %131, %in_4 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %57 : memref<?x32x1xf16>
                  %124 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %60 : memref<?x32x1xf16>
                  %125 = loom.subview %arg1[%27, %110, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %125, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %126 = loom.bufferize_to_tensor %75[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %127 = linalg.fill ins(%cst : f16) outs(%52 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %128 = linalg.batch_matmul ins(%119, %126 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%127 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %75 : memref<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x32x?xf16>
                  %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %arg12, %124 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in_3, %in_4 : f16
                    %132 = arith.addf %in, %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %70 : memref<?x32x32xf16>
                  loom.semaphore_give %51 : memref<?x32x128xf16>
                  %130 = linalg.copy ins(%117 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %54 : memref<?x32x1xf16>
                  scf.yield %130, %123, %129 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %77 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %78 = loom.semaphore_take %77 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %79 = loom.init_tensor %78[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76#1, %76#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%79 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = math.log %in : f16
                  %110 = arith.addf %109, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %47 : memref<?x32x1xf16>
                %81 = loom.broadcast ins(%76#1 : tensor<?x32x1xf16>) outs(%69 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %43 : memref<?x32x1xf16>
                %82 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %83 = loom.semaphore_take %82 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %84 = loom.init_tensor %83[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76#2, %81 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%84 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = arith.divf %in, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %68 : memref<?x32x32xf16>
                loom.semaphore_give %35 : memref<?x32x128xf16>
                %86 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %87 = loom.semaphore_take %86 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %88 = loom.init_tensor %87[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %89 = arith.muli %arg4, %c8 : index
                %90 = arith.addi %89, %c7 : index
                %91 = loom.gather ins(%80 : tensor<?x32x1xf16>) outs(%88 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%89, %arg6], LR : [%90, %arg6]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %78 : memref<?x32x1xf16>
                %92 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %93 = loom.semaphore_take %92 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %94 = loom.init_tensor %93[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %95 = loom.gather ins(%85 : tensor<?x32x128xf16>) outs(%94 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%89, %arg6], LR : [%90, %arg6]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %83 : memref<?x32x128xf16>
                %96 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %98 = loom.init_tensor %97[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %99 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %100 = loom.semaphore_take %99 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %101 = loom.init_tensor %100[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %102 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %104 = loom.init_tensor %103[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %105 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %107 = loom.init_tensor %106[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %108 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %108 {
                  %109 = linalg.fill ins(%cst_1 : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%91 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.maximumf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.subf %in, %in_3 : f16
                    %122 = math.exp %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %87 : memref<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%40 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.divf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %39 : memref<?x32x1xf16>
                  %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%107 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %100 : memref<?x?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%104 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.mulf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x32xf16>
                  loom.semaphore_give %93 : memref<?x?x32x128xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%98 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%116 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x128xf16>
                  %119 = loom.subview %arg2[%27, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %120, %119 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %97 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
