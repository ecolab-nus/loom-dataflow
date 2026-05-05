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
    func.func @flash_decode__x8_y1y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
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
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %33 = arith.addi %arg6, %arg4 : index
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %33], LR : [%c7, %33]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %34 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %35 = arith.muli %28, %21 : index
                %36 = arith.ceildivui %21, %22 : index
                %37 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %38 = loom.semaphore_take %37 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %39 = loom.semaphore_take %37 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %40 = loom.init_tensor %39[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %42 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %48 = loom.init_tensor %47[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %49 = linalg.fill ins(%cst_0 : f16) outs(%48 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %50 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %51 = loom.semaphore_take %50 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %52 = loom.init_tensor %51[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %53 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %54 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %56 = loom.init_tensor %55[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %57 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %59 = loom.init_tensor %58[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %60 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %62 = loom.init_tensor %61[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %63 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %64 = loom.semaphore_take %63 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %65 = loom.init_tensor %64[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %66 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %68 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %69 = loom.semaphore_take %68 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %70 = loom.init_tensor %69[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %71 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %72 = loom.semaphore_take %71 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %71 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.semaphore_take %71 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %77 = loom.init_tensor %76[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %78 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %79 = loom.semaphore_take %78 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %80:3 = scf.for %arg9 = %c0 to %36 step %c1 iter_args(%arg10 = %53, %arg11 = %49, %arg12 = %41) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %112 = arith.muli %arg9, %22 : index
                  %113 = arith.addi %35, %112 : index
                  %114 = loom.subview %arg0[%29, 0, %113] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %114, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %115 = loom.bufferize_to_tensor %67[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %116 = linalg.fill ins(%cst : f16) outs(%70 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %117 = linalg.batch_matmul ins(%34, %115 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%116 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %67 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %118 = linalg.fill ins(%cst_1 : f16) outs(%59 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%117 : tensor<?x32x?xf16>) outs(%118 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %134 = arith.maximumf %in, %out : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%59 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %134 = arith.mulf %in_3, %cst_2 : f16
                    %135 = arith.cmpf ogt, %in, %134 : f16
                    %136 = arith.select %135, %in, %134 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  %121 = loom.broadcast ins(%120 : tensor<?x32x1xf16>) outs(%77 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %121 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%70 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %134 = arith.mulf %in, %cst_2 : f16
                    %135 = arith.subf %134, %in_3 : f16
                    %136 = math.exp %135 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %76 : memref<?x32x32xf16>
                  %123 = linalg.fill ins(%cst : f16) outs(%62 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%122 : tensor<?x32x?xf16>) outs(%123 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %134 = arith.addf %in, %out : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %120 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%65 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %134 = arith.subf %in, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %125, %124 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %134 = arith.mulf %in, %in_3 : f16
                    %135 = arith.addf %134, %in_4 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %61 : memref<?x32x1xf16>
                  %127 = loom.broadcast ins(%125 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %64 : memref<?x32x1xf16>
                  %128 = loom.subview %arg1[%29, %113, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %128, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %129 = loom.bufferize_to_tensor %79[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %130 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %131 = linalg.batch_matmul ins(%122, %129 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%130 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %79 : memref<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x32x?xf16>
                  %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%131, %arg12, %127 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %134 = arith.mulf %in_3, %in_4 : f16
                    %135 = arith.addf %in, %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  loom.semaphore_give %55 : memref<?x32x128xf16>
                  %133 = linalg.copy ins(%120 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %58 : memref<?x32x1xf16>
                  scf.yield %133, %126, %132 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %81 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %82 = loom.semaphore_take %81 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %83 = loom.init_tensor %82[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80#1, %80#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%83 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %112 = math.log %in : f16
                  %113 = arith.addf %112, %in_3 : f16
                  linalg.yield %113 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %51 : memref<?x32x1xf16>
                %85 = loom.broadcast ins(%80#1 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %47 : memref<?x32x1xf16>
                %86 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %87 = loom.semaphore_take %86 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %88 = loom.init_tensor %87[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80#2, %85 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%88 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %112 = arith.divf %in, %in_3 : f16
                  linalg.yield %112 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %72 : memref<?x32x32xf16>
                loom.semaphore_give %39 : memref<?x32x128xf16>
                %90 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %91 = loom.semaphore_take %90 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %92 = loom.init_tensor %91[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %93 = arith.addi %arg6, %arg4 : index
                %94 = loom.gather ins(%84 : tensor<?x32x1xf16>) outs(%92 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %93], LR : [%c7, %93]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %82 : memref<?x32x1xf16>
                %95 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %96 = loom.semaphore_take %95 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %97 = loom.init_tensor %96[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %98 = loom.gather ins(%89 : tensor<?x32x128xf16>) outs(%97 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %93], LR : [%c7, %93]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %87 : memref<?x32x128xf16>
                %99 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %100 = loom.semaphore_take %99 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %101 = loom.init_tensor %100[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %102 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %104 = loom.init_tensor %103[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %105 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %107 = loom.init_tensor %106[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %108 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %109 = loom.semaphore_take %108 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %110 = loom.init_tensor %109[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %111 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %111 {
                  %112 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%94 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %124 = arith.maximumf %in, %out : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %124 = arith.subf %in, %in_3 : f16
                    %125 = math.exp %124 : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %91 : memref<?x?x32x1xf16>
                  loom.semaphore_give %45 : memref<?x32x1xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%114 : tensor<?x?x32x1xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %124 = arith.addf %in, %out : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114, %116 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %124 = arith.divf %in, %in_3 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %118 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%107 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %124 = arith.mulf %in, %in_3 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %109 : memref<?x?x32x32xf16>
                  loom.semaphore_give %96 : memref<?x?x32x128xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%101 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<?x?x32x128xf16>) outs(%120 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %124 = arith.addf %in, %out : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x128xf16>
                  loom.semaphore_give %38 : memref<?x32x128xf16>
                  %122 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %123 = loom.bufferize_to_memref %121 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %123, %122 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %100 : memref<?x32x128xf16>
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
    func.func @flash_decode__x8_y2y4__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc2_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
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
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %33 = arith.muli %arg4, %c2 : index
                %34 = arith.addi %33, %c1 : index
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 2] region : (UL : [%c0, %33], LR : [%c7, %34]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %35 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %36 = arith.muli %28, %21 : index
                %37 = arith.ceildivui %21, %22 : index
                %38 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %39 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %40 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %41 = loom.init_tensor %40[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %42 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %43 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %47 = loom.init_tensor %46[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %48 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %55 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %57 = loom.init_tensor %56[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %66 = loom.init_tensor %65[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %71 = loom.init_tensor %70[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %72 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %73 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %74 = loom.init_tensor %73[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %75 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %76 = loom.init_tensor %75[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %77 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %78 = loom.init_tensor %77[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81:3 = scf.for %arg9 = %c0 to %37 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %42) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %113 = arith.muli %arg9, %22 : index
                  %114 = arith.addi %36, %113 : index
                  %115 = loom.subview %arg0[%29, 0, %114] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %116 = arith.addi %arg6, %33 : index
                  loom.copy %115, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %116], LR : [%arg5, %116]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %117 = loom.bufferize_to_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %119 = linalg.batch_matmul ins(%35, %117 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%118 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %120 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %136 = arith.maximumf %in, %out : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%60 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %136 = arith.mulf %in_3, %cst_2 : f16
                    %137 = arith.cmpf ogt, %in, %136 : f16
                    %138 = arith.select %137, %in, %136 : f16
                    linalg.yield %138 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%78 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119, %123 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%71 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %136 = arith.mulf %in, %cst_2 : f16
                    %137 = arith.subf %136, %in_3 : f16
                    %138 = math.exp %137 : f16
                    linalg.yield %138 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %77 : memref<?x32x32xf16>
                  %125 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%124 : tensor<?x32x?xf16>) outs(%125 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %136 = arith.addf %in, %out : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %122 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%66 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %136 = arith.subf %in, %in_3 : f16
                    %137 = math.exp %136 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x1xf16>
                  %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %127, %126 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %136 = arith.mulf %in, %in_3 : f16
                    %137 = arith.addf %136, %in_4 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %129 = loom.broadcast ins(%127 : tensor<?x32x1xf16>) outs(%76 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %65 : memref<?x32x1xf16>
                  %130 = loom.subview %arg1[%29, %114, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %130, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %116], LR : [%arg5, %116]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %131 = loom.bufferize_to_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %132 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %133 = linalg.batch_matmul ins(%124, %131 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%132 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %70 : memref<?x32x?xf16>
                  %134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%133, %arg12, %129 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %136 = arith.mulf %in_3, %in_4 : f16
                    %137 = arith.addf %in, %136 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %75 : memref<?x32x32xf16>
                  loom.semaphore_give %56 : memref<?x32x128xf16>
                  %135 = linalg.copy ins(%122 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  scf.yield %135, %128, %134 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %82 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %83 = loom.semaphore_take %82 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %84 = loom.init_tensor %83[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#1, %81#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%84 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %113 = math.log %in : f16
                  %114 = arith.addf %113, %in_3 : f16
                  linalg.yield %114 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %52 : memref<?x32x1xf16>
                %86 = loom.broadcast ins(%81#1 : tensor<?x32x1xf16>) outs(%74 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %87 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %88 = loom.semaphore_take %87 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %89 = loom.init_tensor %88[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#2, %86 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%89 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %113 = arith.divf %in, %in_3 : f16
                  linalg.yield %113 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %73 : memref<?x32x32xf16>
                loom.semaphore_give %40 : memref<?x32x128xf16>
                %91 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %92 = loom.semaphore_take %91 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %93 = loom.init_tensor %92[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %94 = arith.addi %arg6, %33 : index
                %95 = loom.gather ins(%85 : tensor<?x32x1xf16>) outs(%93 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %94], LR : [%c7, %94]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %83 : memref<?x32x1xf16>
                %96 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %98 = loom.init_tensor %97[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %99 = loom.gather ins(%90 : tensor<?x32x128xf16>) outs(%98 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %94], LR : [%c7, %94]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %88 : memref<?x32x128xf16>
                %100 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %101 = loom.semaphore_take %100 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %102 = loom.init_tensor %101[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %103 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %104 = loom.semaphore_take %103 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %105 = loom.init_tensor %104[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %106 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %107 = loom.semaphore_take %106 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %108 = loom.init_tensor %107[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %109 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %110 = loom.semaphore_take %109 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %111 = loom.init_tensor %110[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %112 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %112 {
                  %113 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%95 : tensor<?x?x32x1xf16>) outs(%113 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.maximumf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %114 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%105 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.subf %in, %in_3 : f16
                    %126 = math.exp %125 : f16
                    linalg.yield %126 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %92 : memref<?x?x32x1xf16>
                  loom.semaphore_give %46 : memref<?x32x1xf16>
                  %116 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%115 : tensor<?x?x32x1xf16>) outs(%116 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.addf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%115, %117 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%105 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.divf %in, %in_3 : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %44 : memref<?x32x1xf16>
                  %119 = loom.broadcast ins(%118 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %119 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%108 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.mulf %in, %in_3 : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %110 : memref<?x?x32x32xf16>
                  loom.semaphore_give %97 : memref<?x?x32x128xf16>
                  %121 = linalg.fill ins(%cst : f16) outs(%102 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%120 : tensor<?x?x32x128xf16>) outs(%121 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.addf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x128xf16>
                  loom.semaphore_give %39 : memref<?x32x128xf16>
                  %123 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %124 = loom.bufferize_to_memref %122 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %124, %123 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %94], LR : [%arg5, %94]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %101 : memref<?x32x128xf16>
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
    func.func @flash_decode__x8_y4y2__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc4_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c3 = arith.constant 3 : index
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
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %33 = arith.muli %arg4, %c4 : index
                %34 = arith.addi %33, %c3 : index
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %33], LR : [%c7, %34]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %35 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %36 = arith.muli %28, %21 : index
                %37 = arith.ceildivui %21, %22 : index
                %38 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %39 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %40 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %41 = loom.init_tensor %40[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %42 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %43 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %47 = loom.init_tensor %46[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %48 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %55 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %57 = loom.init_tensor %56[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %66 = loom.init_tensor %65[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %71 = loom.init_tensor %70[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %72 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %73 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %74 = loom.init_tensor %73[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %75 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %76 = loom.init_tensor %75[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %77 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %78 = loom.init_tensor %77[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81:3 = scf.for %arg9 = %c0 to %37 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %42) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %113 = arith.muli %arg9, %22 : index
                  %114 = arith.addi %36, %113 : index
                  %115 = loom.subview %arg0[%29, 0, %114] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %116 = arith.addi %arg6, %33 : index
                  loom.copy %115, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %116], LR : [%arg5, %116]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %117 = loom.bufferize_to_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %119 = linalg.batch_matmul ins(%35, %117 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%118 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %120 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %136 = arith.maximumf %in, %out : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%60 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %136 = arith.mulf %in_3, %cst_2 : f16
                    %137 = arith.cmpf ogt, %in, %136 : f16
                    %138 = arith.select %137, %in, %136 : f16
                    linalg.yield %138 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%78 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%119, %123 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%71 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %136 = arith.mulf %in, %cst_2 : f16
                    %137 = arith.subf %136, %in_3 : f16
                    %138 = math.exp %137 : f16
                    linalg.yield %138 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %77 : memref<?x32x32xf16>
                  %125 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%124 : tensor<?x32x?xf16>) outs(%125 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %136 = arith.addf %in, %out : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %122 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%66 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %136 = arith.subf %in, %in_3 : f16
                    %137 = math.exp %136 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x1xf16>
                  %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %127, %126 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %136 = arith.mulf %in, %in_3 : f16
                    %137 = arith.addf %136, %in_4 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %129 = loom.broadcast ins(%127 : tensor<?x32x1xf16>) outs(%76 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %65 : memref<?x32x1xf16>
                  %130 = loom.subview %arg1[%29, %114, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %130, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %116], LR : [%arg5, %116]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %131 = loom.bufferize_to_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %132 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %133 = linalg.batch_matmul ins(%124, %131 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%132 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %70 : memref<?x32x?xf16>
                  %134 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%133, %arg12, %129 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %136 = arith.mulf %in_3, %in_4 : f16
                    %137 = arith.addf %in, %136 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %75 : memref<?x32x32xf16>
                  loom.semaphore_give %56 : memref<?x32x128xf16>
                  %135 = linalg.copy ins(%122 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  scf.yield %135, %128, %134 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %82 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %83 = loom.semaphore_take %82 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %84 = loom.init_tensor %83[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#1, %81#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%84 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %113 = math.log %in : f16
                  %114 = arith.addf %113, %in_3 : f16
                  linalg.yield %114 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %52 : memref<?x32x1xf16>
                %86 = loom.broadcast ins(%81#1 : tensor<?x32x1xf16>) outs(%74 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %87 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %88 = loom.semaphore_take %87 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %89 = loom.init_tensor %88[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#2, %86 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%89 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %113 = arith.divf %in, %in_3 : f16
                  linalg.yield %113 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %73 : memref<?x32x32xf16>
                loom.semaphore_give %40 : memref<?x32x128xf16>
                %91 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %92 = loom.semaphore_take %91 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %93 = loom.init_tensor %92[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %94 = arith.addi %arg6, %33 : index
                %95 = loom.gather ins(%85 : tensor<?x32x1xf16>) outs(%93 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %94], LR : [%c7, %94]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %83 : memref<?x32x1xf16>
                %96 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %98 = loom.init_tensor %97[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %99 = loom.gather ins(%90 : tensor<?x32x128xf16>) outs(%98 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %94], LR : [%c7, %94]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %88 : memref<?x32x128xf16>
                %100 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %101 = loom.semaphore_take %100 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %102 = loom.init_tensor %101[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %103 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %104 = loom.semaphore_take %103 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %105 = loom.init_tensor %104[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %106 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %107 = loom.semaphore_take %106 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %108 = loom.init_tensor %107[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %109 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %110 = loom.semaphore_take %109 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %111 = loom.init_tensor %110[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %112 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %112 {
                  %113 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%95 : tensor<?x?x32x1xf16>) outs(%113 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.maximumf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %114 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%105 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.subf %in, %in_3 : f16
                    %126 = math.exp %125 : f16
                    linalg.yield %126 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %92 : memref<?x?x32x1xf16>
                  loom.semaphore_give %46 : memref<?x32x1xf16>
                  %116 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%115 : tensor<?x?x32x1xf16>) outs(%116 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.addf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%115, %117 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%105 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.divf %in, %in_3 : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %44 : memref<?x32x1xf16>
                  %119 = loom.broadcast ins(%118 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %119 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%108 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.mulf %in, %in_3 : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %110 : memref<?x?x32x32xf16>
                  loom.semaphore_give %97 : memref<?x?x32x128xf16>
                  %121 = linalg.fill ins(%cst : f16) outs(%102 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%120 : tensor<?x?x32x128xf16>) outs(%121 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.addf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x128xf16>
                  loom.semaphore_give %39 : memref<?x32x128xf16>
                  %123 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %124 = loom.bufferize_to_memref %122 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %124, %123 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %94], LR : [%arg5, %94]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %101 : memref<?x32x128xf16>
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
    func.func @flash_decode__x8_y8y1__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
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
                %30 = loom.subview %arg3[%27, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %31 = loom.bufferize_to_tensor %29[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %32 = arith.muli %26, %21 : index
                %33 = arith.ceildivui %21, %22 : index
                %34 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %36 = loom.semaphore_take %34 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %37 = loom.init_tensor %36[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %39 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %41 = loom.init_tensor %40[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %42 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %43 = loom.init_tensor %42[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %44 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = linalg.fill ins(%cst_0 : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %47 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %53 = loom.init_tensor %52[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %54 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %56 = loom.init_tensor %55[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %57 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %59 = loom.init_tensor %58[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %60 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %62 = loom.init_tensor %61[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %63 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %64 = loom.semaphore_take %63 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %65 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %67 = loom.init_tensor %66[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %68 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %69 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %70 = loom.init_tensor %69[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %71 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %72 = loom.init_tensor %71[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %73 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %74 = loom.init_tensor %73[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %75 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %76 = loom.semaphore_take %75 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %77:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %50, %arg11 = %46, %arg12 = %38) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %110 = arith.muli %arg9, %22 : index
                  %111 = arith.addi %32, %110 : index
                  %112 = loom.subview %arg0[%27, 0, %111] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %113 = arith.muli %arg4, %c8 : index
                  %114 = arith.addi %arg6, %113 : index
                  loom.copy %112, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %114], LR : [%arg5, %114]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %115 = loom.bufferize_to_tensor %64[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %116 = linalg.fill ins(%cst : f16) outs(%67 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %117 = linalg.batch_matmul ins(%31, %115 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%116 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %64 : memref<?x128x?xf16>
                  loom.semaphore_give %29 : memref<?x32x128xf16>
                  %118 = linalg.fill ins(%cst_1 : f16) outs(%56 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%117 : tensor<?x32x?xf16>) outs(%118 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %134 = arith.maximumf %in, %out : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%56 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %134 = arith.mulf %in_3, %cst_2 : f16
                    %135 = arith.cmpf ogt, %in, %134 : f16
                    %136 = arith.select %135, %in, %134 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  %121 = loom.broadcast ins(%120 : tensor<?x32x1xf16>) outs(%74 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %121 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%67 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %134 = arith.mulf %in, %cst_2 : f16
                    %135 = arith.subf %134, %in_3 : f16
                    %136 = math.exp %135 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %73 : memref<?x32x32xf16>
                  %123 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%122 : tensor<?x32x?xf16>) outs(%123 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %134 = arith.addf %in, %out : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %120 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%62 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %134 = arith.subf %in, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %125, %124 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %134 = arith.mulf %in, %in_3 : f16
                    %135 = arith.addf %134, %in_4 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %58 : memref<?x32x1xf16>
                  %127 = loom.broadcast ins(%125 : tensor<?x32x1xf16>) outs(%72 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %61 : memref<?x32x1xf16>
                  %128 = loom.subview %arg1[%27, %111, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %128, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %114], LR : [%arg5, %114]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %129 = loom.bufferize_to_tensor %76[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %130 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %131 = linalg.batch_matmul ins(%122, %129 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%130 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %76 : memref<?x?x128xf16>
                  loom.semaphore_give %66 : memref<?x32x?xf16>
                  %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%131, %arg12, %127 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %134 = arith.mulf %in_3, %in_4 : f16
                    %135 = arith.addf %in, %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %71 : memref<?x32x32xf16>
                  loom.semaphore_give %52 : memref<?x32x128xf16>
                  %133 = linalg.copy ins(%120 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %55 : memref<?x32x1xf16>
                  scf.yield %133, %126, %132 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %78 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %79 = loom.semaphore_take %78 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %80 = loom.init_tensor %79[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77#1, %77#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%80 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = math.log %in : f16
                  %111 = arith.addf %110, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %82 = loom.broadcast ins(%77#1 : tensor<?x32x1xf16>) outs(%70 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %44 : memref<?x32x1xf16>
                %83 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %84 = loom.semaphore_take %83 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %85 = loom.init_tensor %84[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77#2, %82 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%85 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = arith.divf %in, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %69 : memref<?x32x32xf16>
                loom.semaphore_give %36 : memref<?x32x128xf16>
                %87 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %88 = loom.semaphore_take %87 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %89 = loom.init_tensor %88[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %90 = arith.muli %arg4, %c8 : index
                %91 = arith.addi %arg6, %90 : index
                %92 = loom.gather ins(%81 : tensor<?x32x1xf16>) outs(%89 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %91], LR : [%c7, %91]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %79 : memref<?x32x1xf16>
                %93 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %95 = loom.init_tensor %94[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %96 = loom.gather ins(%86 : tensor<?x32x128xf16>) outs(%95 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %91], LR : [%c7, %91]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %84 : memref<?x32x128xf16>
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
                  %110 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
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
                  loom.semaphore_give %88 : memref<?x?x32x1xf16>
                  loom.semaphore_give %42 : memref<?x32x1xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
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
                  loom.semaphore_give %40 : memref<?x32x1xf16>
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
                  loom.semaphore_give %35 : memref<?x32x128xf16>
                  %120 = loom.subview %arg2[%27, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %121 = loom.bufferize_to_memref %119 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %121, %120 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %91], LR : [%arg5, %91]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
    func.func @flash_decode__x1x8_y8__d0i1_d1i1_d2i0__f01__dim_y_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
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
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %33 = arith.addi %arg5, %arg4 : index
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%33, %c0], LR : [%33, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %34 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %35 = arith.muli %28, %21 : index
                %36 = arith.ceildivui %21, %22 : index
                %37 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %38 = loom.semaphore_take %37 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %39 = loom.semaphore_take %37 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %40 = loom.init_tensor %39[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %42 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %48 = loom.init_tensor %47[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %49 = linalg.fill ins(%cst_0 : f16) outs(%48 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %50 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %51 = loom.semaphore_take %50 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %52 = loom.init_tensor %51[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %53 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %54 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %56 = loom.init_tensor %55[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %57 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %59 = loom.init_tensor %58[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %60 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %62 = loom.init_tensor %61[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %63 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %64 = loom.semaphore_take %63 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %65 = loom.init_tensor %64[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %66 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %68 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %69 = loom.semaphore_take %68 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %70 = loom.init_tensor %69[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %71 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %72 = loom.semaphore_take %71 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %71 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.semaphore_take %71 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %77 = loom.init_tensor %76[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %78 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %79 = loom.semaphore_take %78 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %80:3 = scf.for %arg9 = %c0 to %36 step %c1 iter_args(%arg10 = %53, %arg11 = %49, %arg12 = %41) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %111 = arith.muli %arg9, %22 : index
                  %112 = arith.addi %35, %111 : index
                  %113 = loom.subview %arg0[%29, 0, %112] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %113, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %114 = loom.bufferize_to_tensor %67[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%70 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %116 = linalg.batch_matmul ins(%34, %114 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%115 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %67 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %117 = linalg.fill ins(%cst_1 : f16) outs(%59 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%59 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in_3, %cst_2 : f16
                    %134 = arith.cmpf ogt, %in, %133 : f16
                    %135 = arith.select %134, %in, %133 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%77 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %120 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%70 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in, %cst_2 : f16
                    %134 = arith.subf %133, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %76 : memref<?x32x32xf16>
                  %122 = linalg.fill ins(%cst : f16) outs(%62 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121 : tensor<?x32x?xf16>) outs(%122 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%65 : tensor<?x32x1xf16>) {
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
                  loom.semaphore_give %61 : memref<?x32x1xf16>
                  %126 = loom.broadcast ins(%124 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %64 : memref<?x32x1xf16>
                  %127 = loom.subview %arg1[%29, %112, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %127, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %128 = loom.bufferize_to_tensor %79[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %129 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %130 = linalg.batch_matmul ins(%121, %128 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%129 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %79 : memref<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x32x?xf16>
                  %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%130, %arg12, %126 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in_3, %in_4 : f16
                    %134 = arith.addf %in, %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  loom.semaphore_give %55 : memref<?x32x128xf16>
                  %132 = linalg.copy ins(%119 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %58 : memref<?x32x1xf16>
                  scf.yield %132, %125, %131 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %81 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %82 = loom.semaphore_take %81 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %83 = loom.init_tensor %82[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80#1, %80#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%83 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = math.log %in : f16
                  %112 = arith.addf %111, %in_3 : f16
                  linalg.yield %112 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %51 : memref<?x32x1xf16>
                %85 = loom.broadcast ins(%80#1 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %47 : memref<?x32x1xf16>
                %86 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %87 = loom.semaphore_take %86 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %88 = loom.init_tensor %87[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80#2, %85 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%88 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %111 = arith.divf %in, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %72 : memref<?x32x32xf16>
                loom.semaphore_give %39 : memref<?x32x128xf16>
                %90 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %91 = loom.semaphore_take %90 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %92 = loom.init_tensor %91[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %93 = loom.gather ins(%84 : tensor<?x32x1xf16>) outs(%92 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%arg4, %arg6], LR : [%arg4, %arg6]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %82 : memref<?x32x1xf16>
                %94 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %95 = loom.semaphore_take %94 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %96 = loom.init_tensor %95[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %97 = loom.gather ins(%89 : tensor<?x32x128xf16>) outs(%96 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%arg4, %arg6], LR : [%arg4, %arg6]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %87 : memref<?x32x128xf16>
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
                  %111 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
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
                  loom.semaphore_give %91 : memref<?x?x32x1xf16>
                  loom.semaphore_give %45 : memref<?x32x1xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
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
                  loom.semaphore_give %43 : memref<?x32x1xf16>
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
                  loom.semaphore_give %38 : memref<?x32x128xf16>
                  %121 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %122 = loom.bufferize_to_memref %120 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %122, %121 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
    func.func @flash_decode__x2x4_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc2_dim_y_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
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
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %33 = arith.muli %arg4, %c2 : index
                %34 = arith.addi %33, %c1 : index
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 8] region : (UL : [%33, %c0], LR : [%34, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %35 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %36 = arith.muli %28, %21 : index
                %37 = arith.ceildivui %21, %22 : index
                %38 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %39 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %40 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %41 = loom.init_tensor %40[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %42 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %43 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %47 = loom.init_tensor %46[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %48 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %55 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %57 = loom.init_tensor %56[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %66 = loom.init_tensor %65[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %71 = loom.init_tensor %70[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %72 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %73 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %74 = loom.init_tensor %73[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %75 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %76 = loom.init_tensor %75[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %77 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %78 = loom.init_tensor %77[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81:3 = scf.for %arg9 = %c0 to %37 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %42) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %112 = arith.muli %arg9, %22 : index
                  %113 = arith.addi %36, %112 : index
                  %114 = loom.subview %arg0[%29, 0, %113] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %115 = arith.addi %arg5, %33 : index
                  loom.copy %114, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%115, %arg6], LR : [%115, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %116 = loom.bufferize_to_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %118 = linalg.batch_matmul ins(%35, %116 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%117 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %119 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%118 : tensor<?x32x?xf16>) outs(%119 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %135 = arith.maximumf %in, %out : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %120 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%60 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %135 = arith.mulf %in_3, %cst_2 : f16
                    %136 = arith.cmpf ogt, %in, %135 : f16
                    %137 = arith.select %136, %in, %135 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = loom.broadcast ins(%121 : tensor<?x32x1xf16>) outs(%78 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%118, %122 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%71 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %135 = arith.mulf %in, %cst_2 : f16
                    %136 = arith.subf %135, %in_3 : f16
                    %137 = math.exp %136 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %77 : memref<?x32x32xf16>
                  %124 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%123 : tensor<?x32x?xf16>) outs(%124 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %135 = arith.addf %in, %out : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%66 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %135 = arith.subf %in, %in_3 : f16
                    %136 = math.exp %135 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %126, %125 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %135 = arith.mulf %in, %in_3 : f16
                    %136 = arith.addf %135, %in_4 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %128 = loom.broadcast ins(%126 : tensor<?x32x1xf16>) outs(%76 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %65 : memref<?x32x1xf16>
                  %129 = loom.subview %arg1[%29, %113, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %129, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%115, %arg6], LR : [%115, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %130 = loom.bufferize_to_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %131 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %132 = linalg.batch_matmul ins(%123, %130 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%131 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %70 : memref<?x32x?xf16>
                  %133 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%132, %arg12, %128 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %135 = arith.mulf %in_3, %in_4 : f16
                    %136 = arith.addf %in, %135 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %75 : memref<?x32x32xf16>
                  loom.semaphore_give %56 : memref<?x32x128xf16>
                  %134 = linalg.copy ins(%121 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  scf.yield %134, %127, %133 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %82 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %83 = loom.semaphore_take %82 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %84 = loom.init_tensor %83[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#1, %81#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%84 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %112 = math.log %in : f16
                  %113 = arith.addf %112, %in_3 : f16
                  linalg.yield %113 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %52 : memref<?x32x1xf16>
                %86 = loom.broadcast ins(%81#1 : tensor<?x32x1xf16>) outs(%74 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %87 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %88 = loom.semaphore_take %87 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %89 = loom.init_tensor %88[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#2, %86 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%89 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %112 = arith.divf %in, %in_3 : f16
                  linalg.yield %112 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %73 : memref<?x32x32xf16>
                loom.semaphore_give %40 : memref<?x32x128xf16>
                %91 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %92 = loom.semaphore_take %91 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %93 = loom.init_tensor %92[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %94 = loom.gather ins(%85 : tensor<?x32x1xf16>) outs(%93 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%33, %arg6], LR : [%34, %arg6]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %83 : memref<?x32x1xf16>
                %95 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %96 = loom.semaphore_take %95 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %97 = loom.init_tensor %96[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %98 = loom.gather ins(%90 : tensor<?x32x128xf16>) outs(%97 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%33, %arg6], LR : [%34, %arg6]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %88 : memref<?x32x128xf16>
                %99 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %100 = loom.semaphore_take %99 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %101 = loom.init_tensor %100[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %102 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %104 = loom.init_tensor %103[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %105 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %107 = loom.init_tensor %106[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %108 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %109 = loom.semaphore_take %108 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %110 = loom.init_tensor %109[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %111 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %111 {
                  %112 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%94 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.maximumf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.subf %in, %in_3 : f16
                    %126 = math.exp %125 : f16
                    linalg.yield %126 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %92 : memref<?x?x32x1xf16>
                  loom.semaphore_give %46 : memref<?x32x1xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%114 : tensor<?x?x32x1xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.addf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114, %116 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.divf %in, %in_3 : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %44 : memref<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %118 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%107 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.mulf %in, %in_3 : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %109 : memref<?x?x32x32xf16>
                  loom.semaphore_give %96 : memref<?x?x32x128xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%101 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<?x?x32x128xf16>) outs(%120 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.addf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x128xf16>
                  loom.semaphore_give %39 : memref<?x32x128xf16>
                  %122 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %123 = loom.bufferize_to_memref %121 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %124 = arith.addi %arg5, %33 : index
                  loom.copy %123, %122 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%124, %arg6], LR : [%124, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %100 : memref<?x32x128xf16>
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
    func.func @flash_decode__x4x2_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc4_dim_y_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
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
                %32 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %33 = arith.muli %arg4, %c4 : index
                %34 = arith.addi %33, %c3 : index
                loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%33, %c0], LR : [%34, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %35 = loom.bufferize_to_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %36 = arith.muli %28, %21 : index
                %37 = arith.ceildivui %21, %22 : index
                %38 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %39 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %40 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %41 = loom.init_tensor %40[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %42 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %43 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %47 = loom.init_tensor %46[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %48 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %55 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %57 = loom.init_tensor %56[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %58 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %66 = loom.init_tensor %65[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %71 = loom.init_tensor %70[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %72 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %73 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %74 = loom.init_tensor %73[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %75 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %76 = loom.init_tensor %75[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %77 = loom.semaphore_take %72 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %78 = loom.init_tensor %77[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81:3 = scf.for %arg9 = %c0 to %37 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %42) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %112 = arith.muli %arg9, %22 : index
                  %113 = arith.addi %36, %112 : index
                  %114 = loom.subview %arg0[%29, 0, %113] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %115 = arith.addi %arg5, %33 : index
                  loom.copy %114, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%115, %arg6], LR : [%115, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %116 = loom.bufferize_to_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %118 = linalg.batch_matmul ins(%35, %116 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%117 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  loom.semaphore_give %31 : memref<?x32x128xf16>
                  %119 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%118 : tensor<?x32x?xf16>) outs(%119 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %135 = arith.maximumf %in, %out : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %120 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%60 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %135 = arith.mulf %in_3, %cst_2 : f16
                    %136 = arith.cmpf ogt, %in, %135 : f16
                    %137 = arith.select %136, %in, %135 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = loom.broadcast ins(%121 : tensor<?x32x1xf16>) outs(%78 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%118, %122 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%71 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %135 = arith.mulf %in, %cst_2 : f16
                    %136 = arith.subf %135, %in_3 : f16
                    %137 = math.exp %136 : f16
                    linalg.yield %137 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %77 : memref<?x32x32xf16>
                  %124 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%123 : tensor<?x32x?xf16>) outs(%124 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %135 = arith.addf %in, %out : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%66 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %135 = arith.subf %in, %in_3 : f16
                    %136 = math.exp %135 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %126, %125 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %135 = arith.mulf %in, %in_3 : f16
                    %136 = arith.addf %135, %in_4 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %128 = loom.broadcast ins(%126 : tensor<?x32x1xf16>) outs(%76 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %65 : memref<?x32x1xf16>
                  %129 = loom.subview %arg1[%29, %113, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %129, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%115, %arg6], LR : [%115, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %130 = loom.bufferize_to_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %131 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %132 = linalg.batch_matmul ins(%123, %130 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%131 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %70 : memref<?x32x?xf16>
                  %133 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%132, %arg12, %128 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %135 = arith.mulf %in_3, %in_4 : f16
                    %136 = arith.addf %in, %135 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %75 : memref<?x32x32xf16>
                  loom.semaphore_give %56 : memref<?x32x128xf16>
                  %134 = linalg.copy ins(%121 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  scf.yield %134, %127, %133 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %82 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %83 = loom.semaphore_take %82 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %84 = loom.init_tensor %83[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#1, %81#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%84 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %112 = math.log %in : f16
                  %113 = arith.addf %112, %in_3 : f16
                  linalg.yield %113 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %52 : memref<?x32x1xf16>
                %86 = loom.broadcast ins(%81#1 : tensor<?x32x1xf16>) outs(%74 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %87 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %88 = loom.semaphore_take %87 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %89 = loom.init_tensor %88[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#2, %86 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%89 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %112 = arith.divf %in, %in_3 : f16
                  linalg.yield %112 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %73 : memref<?x32x32xf16>
                loom.semaphore_give %40 : memref<?x32x128xf16>
                %91 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %92 = loom.semaphore_take %91 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %93 = loom.init_tensor %92[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %94 = loom.gather ins(%85 : tensor<?x32x1xf16>) outs(%93 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%33, %arg6], LR : [%34, %arg6]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %83 : memref<?x32x1xf16>
                %95 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %96 = loom.semaphore_take %95 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %97 = loom.init_tensor %96[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %98 = loom.gather ins(%90 : tensor<?x32x128xf16>) outs(%97 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%33, %arg6], LR : [%34, %arg6]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %88 : memref<?x32x128xf16>
                %99 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %100 = loom.semaphore_take %99 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %101 = loom.init_tensor %100[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %102 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %104 = loom.init_tensor %103[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %105 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %107 = loom.init_tensor %106[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %108 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %109 = loom.semaphore_take %108 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %110 = loom.init_tensor %109[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                %111 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %111 {
                  %112 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%94 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.maximumf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.subf %in, %in_3 : f16
                    %126 = math.exp %125 : f16
                    linalg.yield %126 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %92 : memref<?x?x32x1xf16>
                  loom.semaphore_give %46 : memref<?x32x1xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%114 : tensor<?x?x32x1xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.addf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%114, %116 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.divf %in, %in_3 : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %44 : memref<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %118 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%107 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %125 = arith.mulf %in, %in_3 : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %109 : memref<?x?x32x32xf16>
                  loom.semaphore_give %96 : memref<?x?x32x128xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%101 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<?x?x32x128xf16>) outs(%120 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %125 = arith.addf %in, %out : f16
                    linalg.yield %125 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x128xf16>
                  loom.semaphore_give %39 : memref<?x32x128xf16>
                  %122 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %123 = loom.bufferize_to_memref %121 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %124 = arith.addi %arg5, %33 : index
                  loom.copy %123, %122 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%124, %arg6], LR : [%124, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %100 : memref<?x32x128xf16>
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
    func.func @flash_decode__x8x1_y8__d0i1_d1i1_d2i0__f01__dim_x_level1_bc8_dim_y_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
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
                %30 = loom.subview %arg3[%27, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %31 = loom.bufferize_to_tensor %29[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %32 = arith.muli %26, %21 : index
                %33 = arith.ceildivui %21, %22 : index
                %34 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %36 = loom.semaphore_take %34 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %37 = loom.init_tensor %36[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %39 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %41 = loom.init_tensor %40[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %42 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %43 = loom.init_tensor %42[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %44 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = linalg.fill ins(%cst_0 : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %47 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %53 = loom.init_tensor %52[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %54 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %56 = loom.init_tensor %55[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %57 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %59 = loom.init_tensor %58[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %60 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %62 = loom.init_tensor %61[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %63 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %64 = loom.semaphore_take %63 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %65 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %67 = loom.init_tensor %66[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %68 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
                %69 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %70 = loom.init_tensor %69[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %71 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %72 = loom.init_tensor %71[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %73 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %74 = loom.init_tensor %73[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %75 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %76 = loom.semaphore_take %75 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %77:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %50, %arg11 = %46, %arg12 = %38) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %110 = arith.muli %arg9, %22 : index
                  %111 = arith.addi %32, %110 : index
                  %112 = loom.subview %arg0[%27, 0, %111] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %113 = arith.muli %arg4, %c8 : index
                  %114 = arith.addi %arg5, %113 : index
                  loom.copy %112, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%114, %arg6], LR : [%114, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %115 = loom.bufferize_to_tensor %64[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %116 = linalg.fill ins(%cst : f16) outs(%67 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %117 = linalg.batch_matmul ins(%31, %115 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%116 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %64 : memref<?x128x?xf16>
                  loom.semaphore_give %29 : memref<?x32x128xf16>
                  %118 = linalg.fill ins(%cst_1 : f16) outs(%56 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%117 : tensor<?x32x?xf16>) outs(%118 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %134 = arith.maximumf %in, %out : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%56 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %134 = arith.mulf %in_3, %cst_2 : f16
                    %135 = arith.cmpf ogt, %in, %134 : f16
                    %136 = arith.select %135, %in, %134 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x1xf16>
                  %121 = loom.broadcast ins(%120 : tensor<?x32x1xf16>) outs(%74 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%117, %121 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%67 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %134 = arith.mulf %in, %cst_2 : f16
                    %135 = arith.subf %134, %in_3 : f16
                    %136 = math.exp %135 : f16
                    linalg.yield %136 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %73 : memref<?x32x32xf16>
                  %123 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%122 : tensor<?x32x?xf16>) outs(%123 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %134 = arith.addf %in, %out : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %120 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%62 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %134 = arith.subf %in, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %125, %124 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %134 = arith.mulf %in, %in_3 : f16
                    %135 = arith.addf %134, %in_4 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %58 : memref<?x32x1xf16>
                  %127 = loom.broadcast ins(%125 : tensor<?x32x1xf16>) outs(%72 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %61 : memref<?x32x1xf16>
                  %128 = loom.subview %arg1[%27, %111, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %128, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%114, %arg6], LR : [%114, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %129 = loom.bufferize_to_tensor %76[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %130 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %131 = linalg.batch_matmul ins(%122, %129 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%130 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %76 : memref<?x?x128xf16>
                  loom.semaphore_give %66 : memref<?x32x?xf16>
                  %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%131, %arg12, %127 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %134 = arith.mulf %in_3, %in_4 : f16
                    %135 = arith.addf %in, %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %71 : memref<?x32x32xf16>
                  loom.semaphore_give %52 : memref<?x32x128xf16>
                  %133 = linalg.copy ins(%120 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %55 : memref<?x32x1xf16>
                  scf.yield %133, %126, %132 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %78 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
                %79 = loom.semaphore_take %78 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %80 = loom.init_tensor %79[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77#1, %77#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%80 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = math.log %in : f16
                  %111 = arith.addf %110, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %82 = loom.broadcast ins(%77#1 : tensor<?x32x1xf16>) outs(%70 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %44 : memref<?x32x1xf16>
                %83 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
                %84 = loom.semaphore_take %83 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %85 = loom.init_tensor %84[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77#2, %82 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%85 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = arith.divf %in, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %69 : memref<?x32x32xf16>
                loom.semaphore_give %36 : memref<?x32x128xf16>
                %87 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %88 = loom.semaphore_take %87 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %89 = loom.init_tensor %88[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %90 = arith.muli %arg4, %c8 : index
                %91 = arith.addi %90, %c7 : index
                %92 = loom.gather ins(%81 : tensor<?x32x1xf16>) outs(%89 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%90, %arg6], LR : [%91, %arg6]) -> tensor<?x?x32x1xf16>
                loom.semaphore_give %79 : memref<?x32x1xf16>
                %93 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %95 = loom.init_tensor %94[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %96 = loom.gather ins(%86 : tensor<?x32x128xf16>) outs(%95 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%90, %arg6], LR : [%91, %arg6]) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %84 : memref<?x32x128xf16>
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
                  %110 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%92 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.maximumf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%92, %111 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.subf %in, %in_3 : f16
                    %124 = math.exp %123 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %88 : memref<?x?x32x1xf16>
                  loom.semaphore_give %42 : memref<?x32x1xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%112 : tensor<?x?x32x1xf16>) outs(%113 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %114 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.divf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %40 : memref<?x32x1xf16>
                  %116 = loom.broadcast ins(%115 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %101 : memref<?x?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%96, %116 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%105 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.mulf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x32xf16>
                  loom.semaphore_give %94 : memref<?x?x32x128xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%99 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%117 : tensor<?x?x32x128xf16>) outs(%118 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x128xf16>
                  loom.semaphore_give %35 : memref<?x32x128xf16>
                  %120 = loom.subview %arg2[%27, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %121 = loom.bufferize_to_memref %119 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %122 = arith.addi %arg5, %90 : index
                  loom.copy %121, %120 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%122, %arg6], LR : [%122, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %98 : memref<?x32x128xf16>
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
