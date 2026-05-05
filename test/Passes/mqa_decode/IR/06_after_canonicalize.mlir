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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y1y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (1) {
            scf.for %arg7 = %c0 to %c2 step %c1 {
              scf.for %arg8 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %c0_3 = arith.constant 0 : index
                %c0_4 = arith.constant 0 : index
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_3, %c0_4)
                %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %25 = arith.addi %arg6, %arg4 : index
                loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %25], LR : [%c7, %25]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
                %26 = loom.bufferize_to_tensor %23[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %27 = arith.muli %21, %c512 : index
                %28 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %30 = loom.init_tensor %29[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %32 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %33 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %34 = loom.init_tensor %33[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %35 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %37 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %39 = linalg.fill ins(%cst_0 : f16) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %42 = loom.init_tensor %41[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %43 = linalg.fill ins(%cst_1 : f16) outs(%42 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %44 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %45 = loom.semaphore_take %44 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %46 = loom.init_tensor %45[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %47 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %48 = loom.semaphore_take %47 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %49 = loom.init_tensor %48[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %50 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %51 = loom.semaphore_take %50 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %52 = loom.init_tensor %51[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %53 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %54 = loom.semaphore_take %53 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %55 = loom.init_tensor %54[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %56 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
                %57 = loom.semaphore_take %56 : memref<1x128x512xf16> -> memref<1x128x512xf16>
                %58 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %59 = loom.semaphore_take %58 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %60 = loom.init_tensor %59[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %61 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
                %62 = loom.semaphore_take %61 : memref<1x32x32xf16> -> memref<1x32x32xf16>
                %63 = loom.init_tensor %62[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
                %64 = loom.semaphore_take %61 : memref<1x32x32xf16> -> memref<1x32x32xf16>
                %65 = loom.init_tensor %64[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
                %66 = loom.semaphore_take %61 : memref<1x32x32xf16> -> memref<1x32x32xf16>
                %67 = loom.init_tensor %66[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
                %68 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
                %69 = loom.semaphore_take %68 : memref<1x512x128xf16> -> memref<1x512x128xf16>
                %c0_5 = arith.constant 0 : index
                %70 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %27)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%70], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %reinterpret_cast_6, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
                %71 = loom.bufferize_to_tensor %57[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %72 = linalg.fill ins(%cst : f16) outs(%60 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                %73 = linalg.batch_matmul ins(%26, %71 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%72 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                loom.semaphore_give %57 : memref<1x128x512xf16>
                loom.semaphore_give %23 : memref<1x32x128xf16>
                %74 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%73 : tensor<1x32x512xf16>) outs(%74 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %121 = arith.maximumf %in, %out : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%49 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = arith.mulf %in_9, %cst_2 : f16
                  %122 = arith.cmpf ogt, %in, %121 : f16
                  %123 = arith.select %122, %in, %121 : f16
                  linalg.yield %123 : f16
                } -> tensor<1x32x1xf16>
                %77 = loom.broadcast ins(%76 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %77 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%60 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = arith.mulf %in, %cst_2 : f16
                  %122 = arith.subf %121, %in_9 : f16
                  %123 = math.exp %122 : f16
                  linalg.yield %123 : f16
                } -> tensor<1x32x512xf16>
                loom.semaphore_give %66 : memref<1x32x32xf16>
                %79 = linalg.fill ins(%cst : f16) outs(%52 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x32x512xf16>) outs(%79 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %121 = arith.addf %in, %out : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%55 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = arith.subf %in, %in_9 : f16
                  %122 = math.exp %121 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %81, %80 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%39 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %121 = arith.mulf %in, %in_9 : f16
                  %122 = arith.addf %121, %in_10 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %51 : memref<1x32x1xf16>
                %83 = loom.broadcast ins(%81 : tensor<1x32x1xf16>) outs(%65 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %54 : memref<1x32x1xf16>
                %c0_7 = arith.constant 0 : index
                %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %27, %c0_7)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_8, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %85 = loom.bufferize_to_tensor %69[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %86 = linalg.fill ins(%cst : f16) outs(%46 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %87 = linalg.batch_matmul ins(%78, %85 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                loom.semaphore_give %69 : memref<1x512x128xf16>
                loom.semaphore_give %59 : memref<1x32x512xf16>
                %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %31, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%31 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %121 = arith.mulf %in_9, %in_10 : f16
                  %122 = arith.addf %in, %121 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %64 : memref<1x32x32xf16>
                loom.semaphore_give %45 : memref<1x32x128xf16>
                %89 = linalg.copy ins(%76 : tensor<1x32x1xf16>) outs(%42 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                loom.semaphore_give %48 : memref<1x32x1xf16>
                %90 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %91 = loom.semaphore_take %90 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %92 = loom.init_tensor %91[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %89 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%92 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = math.log %in : f16
                  %122 = arith.addf %121, %in_9 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %41 : memref<1x32x1xf16>
                %94 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%63 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %37 : memref<1x32x1xf16>
                %95 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %96 = loom.semaphore_take %95 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %97 = loom.init_tensor %96[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %94 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%97 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = arith.divf %in, %in_9 : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %62 : memref<1x32x32xf16>
                loom.semaphore_give %29 : memref<1x32x128xf16>
                %99 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %100 = loom.semaphore_take %99 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %101 = loom.init_tensor %100[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %102 = arith.addi %arg6, %arg4 : index
                %103 = loom.gather ins(%93 : tensor<1x32x1xf16>) outs(%101 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %102], LR : [%c7, %102]) -> tensor<16x1x32x1xf16>
                loom.semaphore_give %91 : memref<1x32x1xf16>
                %104 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %105 = loom.semaphore_take %104 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %106 = loom.init_tensor %105[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %107 = loom.gather ins(%98 : tensor<1x32x128xf16>) outs(%106 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %102], LR : [%c7, %102]) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %96 : memref<1x32x128xf16>
                %108 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %109 = loom.semaphore_take %108 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %110 = loom.init_tensor %109[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %111 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %112 = loom.semaphore_take %111 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %113 = loom.init_tensor %112[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %114 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %115 = loom.semaphore_take %114 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %116 = loom.init_tensor %115[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %117 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
                %118 = loom.semaphore_take %117 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
                %119 = loom.init_tensor %118[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
                %120 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %120 {
                  %121 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<16x1x32x1xf16>) outs(%121 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<1x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %122 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %133 = arith.subf %in, %in_12 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %100 : memref<16x1x32x1xf16>
                  loom.semaphore_give %35 : memref<1x32x1xf16>
                  %124 = linalg.fill ins(%cst : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%123 : tensor<16x1x32x1xf16>) outs(%124 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<1x32x1xf16>
                  %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%123, %125 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %133 = arith.divf %in, %in_12 : f16
                    linalg.yield %133 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %33 : memref<1x32x1xf16>
                  %127 = loom.broadcast ins(%126 : tensor<16x1x32x1xf16>) outs(%119 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %112 : memref<16x1x32x1xf16>
                  %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %127 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%116 : tensor<16x1x32x128xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %133 = arith.mulf %in, %in_12 : f16
                    linalg.yield %133 : f16
                  } -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %118 : memref<16x1x32x32xf16>
                  loom.semaphore_give %105 : memref<16x1x32x128xf16>
                  %129 = linalg.fill ins(%cst : f16) outs(%110 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                  %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%128 : tensor<16x1x32x128xf16>) outs(%129 : tensor<1x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<1x32x128xf16>
                  loom.semaphore_give %115 : memref<16x1x32x128xf16>
                  %c0_9 = arith.constant 0 : index
                  %c0_10 = arith.constant 0 : index
                  %131 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%131], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %132 = loom.bufferize_to_memref %130 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                  loom.copy %132, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %109 : memref<1x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y2y4__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc2_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (2) {
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %25 = arith.muli %arg4, %c2 : index
              %26 = arith.addi %25, %c1 : index
              loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 2] region : (UL : [%c0, %25], LR : [%c7, %26]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %27 = loom.bufferize_to_tensor %23[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %28 = arith.muli %21, %c512 : index
              %29 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %30 = loom.semaphore_take %29 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %31 = loom.init_tensor %30[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %35 = loom.init_tensor %34[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %36 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %37 = loom.init_tensor %36[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %38 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %39 = loom.init_tensor %38[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %40 = linalg.fill ins(%cst_0 : f16) outs(%39 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %43 = loom.init_tensor %42[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %44 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %45 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %46 = loom.semaphore_take %45 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %47 = loom.init_tensor %46[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %48 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %50 = loom.init_tensor %49[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %56 = loom.init_tensor %55[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %57 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %58 = loom.semaphore_take %57 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %59 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %60 = loom.semaphore_take %59 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %61 = loom.init_tensor %60[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %62 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %64 = loom.init_tensor %63[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %65 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %66 = loom.init_tensor %65[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %67 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %68 = loom.init_tensor %67[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %69 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %70 = loom.semaphore_take %69 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %28)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %72 = arith.addi %arg6, %25 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %72], LR : [%arg5, %72]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %73 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %74 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %75 = linalg.batch_matmul ins(%27, %73 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%74 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              loom.semaphore_give %23 : memref<1x32x128xf16>
              %76 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%75 : tensor<1x32x512xf16>) outs(%76 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.maximumf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in_9, %cst_2 : f16
                %124 = arith.cmpf ogt, %in, %123 : f16
                %125 = arith.select %124, %in, %123 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              %79 = loom.broadcast ins(%78 : tensor<1x32x1xf16>) outs(%68 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %79 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in, %cst_2 : f16
                %124 = arith.subf %123, %in_9 : f16
                %125 = math.exp %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %67 : memref<1x32x32xf16>
              %81 = linalg.fill ins(%cst : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%80 : tensor<1x32x512xf16>) outs(%81 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.addf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.subf %in, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %83, %82 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in, %in_9 : f16
                %124 = arith.addf %123, %in_10 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %85 = loom.broadcast ins(%83 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %28, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %72], LR : [%arg5, %72]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %87 = loom.bufferize_to_tensor %70[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %88 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %89 = linalg.batch_matmul ins(%80, %87 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%88 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %70 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %32, %85 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%32 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in_9, %in_10 : f16
                %124 = arith.addf %in, %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %65 : memref<1x32x32xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %91 = linalg.copy ins(%78 : tensor<1x32x1xf16>) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              %92 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %93 = loom.semaphore_take %92 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %94 = loom.init_tensor %93[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %91 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%94 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = math.log %in : f16
                %124 = arith.addf %123, %in_9 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %42 : memref<1x32x1xf16>
              %96 = loom.broadcast ins(%84 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %38 : memref<1x32x1xf16>
              %97 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %98 = loom.semaphore_take %97 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %99 = loom.init_tensor %98[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %96 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%99 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.divf %in, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %63 : memref<1x32x32xf16>
              loom.semaphore_give %30 : memref<1x32x128xf16>
              %101 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %102 = loom.semaphore_take %101 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %103 = loom.init_tensor %102[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = arith.addi %arg6, %25 : index
              %105 = loom.gather ins(%95 : tensor<1x32x1xf16>) outs(%103 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %104], LR : [%c7, %104]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %93 : memref<1x32x1xf16>
              %106 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %107 = loom.semaphore_take %106 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %108 = loom.init_tensor %107[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %109 = loom.gather ins(%100 : tensor<1x32x128xf16>) outs(%108 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %104], LR : [%c7, %104]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %98 : memref<1x32x128xf16>
              %110 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %111 = loom.semaphore_take %110 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %112 = loom.init_tensor %111[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %113 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %114 = loom.semaphore_take %113 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %115 = loom.init_tensor %114[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %116 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %117 = loom.semaphore_take %116 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %118 = loom.init_tensor %117[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %119 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
              %120 = loom.semaphore_take %119 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
              %121 = loom.init_tensor %120[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
              %122 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %122 {
                %123 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%105 : tensor<16x1x32x1xf16>) outs(%123 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.maximumf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %124 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.subf %in, %in_12 : f16
                  %136 = math.exp %135 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %102 : memref<16x1x32x1xf16>
                loom.semaphore_give %36 : memref<1x32x1xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x1xf16>) outs(%126 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125, %127 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.divf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %129 = loom.broadcast ins(%128 : tensor<16x1x32x1xf16>) outs(%121 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %114 : memref<16x1x32x1xf16>
                %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %129 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%118 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.mulf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %120 : memref<16x1x32x32xf16>
                loom.semaphore_give %107 : memref<16x1x32x128xf16>
                %131 = linalg.fill ins(%cst : f16) outs(%112 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%130 : tensor<16x1x32x128xf16>) outs(%131 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %117 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %133 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%133], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %134 = loom.bufferize_to_memref %132 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                loom.copy %134, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %104], LR : [%arg5, %104]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %111 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y4y2__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc4_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c8 = arith.constant 8 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (4) {
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %25 = arith.muli %arg4, %c4 : index
              %26 = arith.addi %25, %c3 : index
              loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %25], LR : [%c7, %26]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %27 = loom.bufferize_to_tensor %23[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %28 = arith.muli %21, %c512 : index
              %29 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %30 = loom.semaphore_take %29 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %31 = loom.init_tensor %30[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %35 = loom.init_tensor %34[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %36 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %37 = loom.init_tensor %36[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %38 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %39 = loom.init_tensor %38[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %40 = linalg.fill ins(%cst_0 : f16) outs(%39 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %43 = loom.init_tensor %42[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %44 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %45 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %46 = loom.semaphore_take %45 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %47 = loom.init_tensor %46[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %48 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %50 = loom.init_tensor %49[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %56 = loom.init_tensor %55[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %57 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %58 = loom.semaphore_take %57 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %59 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %60 = loom.semaphore_take %59 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %61 = loom.init_tensor %60[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %62 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %64 = loom.init_tensor %63[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %65 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %66 = loom.init_tensor %65[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %67 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %68 = loom.init_tensor %67[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %69 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %70 = loom.semaphore_take %69 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %28)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %72 = arith.addi %arg6, %25 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %72], LR : [%arg5, %72]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %73 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %74 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %75 = linalg.batch_matmul ins(%27, %73 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%74 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              loom.semaphore_give %23 : memref<1x32x128xf16>
              %76 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%75 : tensor<1x32x512xf16>) outs(%76 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.maximumf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in_9, %cst_2 : f16
                %124 = arith.cmpf ogt, %in, %123 : f16
                %125 = arith.select %124, %in, %123 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              %79 = loom.broadcast ins(%78 : tensor<1x32x1xf16>) outs(%68 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %79 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in, %cst_2 : f16
                %124 = arith.subf %123, %in_9 : f16
                %125 = math.exp %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %67 : memref<1x32x32xf16>
              %81 = linalg.fill ins(%cst : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%80 : tensor<1x32x512xf16>) outs(%81 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.addf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.subf %in, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %83, %82 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in, %in_9 : f16
                %124 = arith.addf %123, %in_10 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %85 = loom.broadcast ins(%83 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %28, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %72], LR : [%arg5, %72]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %87 = loom.bufferize_to_tensor %70[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %88 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %89 = linalg.batch_matmul ins(%80, %87 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%88 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %70 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %32, %85 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%32 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in_9, %in_10 : f16
                %124 = arith.addf %in, %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %65 : memref<1x32x32xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %91 = linalg.copy ins(%78 : tensor<1x32x1xf16>) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              %92 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %93 = loom.semaphore_take %92 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %94 = loom.init_tensor %93[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %91 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%94 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = math.log %in : f16
                %124 = arith.addf %123, %in_9 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %42 : memref<1x32x1xf16>
              %96 = loom.broadcast ins(%84 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %38 : memref<1x32x1xf16>
              %97 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %98 = loom.semaphore_take %97 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %99 = loom.init_tensor %98[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %96 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%99 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.divf %in, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %63 : memref<1x32x32xf16>
              loom.semaphore_give %30 : memref<1x32x128xf16>
              %101 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %102 = loom.semaphore_take %101 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %103 = loom.init_tensor %102[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = arith.addi %arg6, %25 : index
              %105 = loom.gather ins(%95 : tensor<1x32x1xf16>) outs(%103 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %104], LR : [%c7, %104]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %93 : memref<1x32x1xf16>
              %106 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %107 = loom.semaphore_take %106 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %108 = loom.init_tensor %107[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %109 = loom.gather ins(%100 : tensor<1x32x128xf16>) outs(%108 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %104], LR : [%c7, %104]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %98 : memref<1x32x128xf16>
              %110 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %111 = loom.semaphore_take %110 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %112 = loom.init_tensor %111[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %113 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %114 = loom.semaphore_take %113 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %115 = loom.init_tensor %114[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %116 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %117 = loom.semaphore_take %116 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %118 = loom.init_tensor %117[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %119 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
              %120 = loom.semaphore_take %119 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
              %121 = loom.init_tensor %120[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
              %122 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %122 {
                %123 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%105 : tensor<16x1x32x1xf16>) outs(%123 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.maximumf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %124 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.subf %in, %in_12 : f16
                  %136 = math.exp %135 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %102 : memref<16x1x32x1xf16>
                loom.semaphore_give %36 : memref<1x32x1xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x1xf16>) outs(%126 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125, %127 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.divf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %129 = loom.broadcast ins(%128 : tensor<16x1x32x1xf16>) outs(%121 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %114 : memref<16x1x32x1xf16>
                %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %129 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%118 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.mulf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %120 : memref<16x1x32x32xf16>
                loom.semaphore_give %107 : memref<16x1x32x128xf16>
                %131 = linalg.fill ins(%cst : f16) outs(%112 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%130 : tensor<16x1x32x128xf16>) outs(%131 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %117 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %133 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%133], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %134 = loom.bufferize_to_memref %132 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                loom.copy %134, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %104], LR : [%arg5, %104]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %111 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y8y1__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c16 = arith.constant 16 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c16 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %21 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %22 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %24 = loom.bufferize_to_tensor %22[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %25 = arith.muli %20, %c512 : index
              %26 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %27 = loom.semaphore_take %26 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %28 = loom.init_tensor %27[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %30 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %31 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %32 = loom.init_tensor %31[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %33 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %34 = loom.init_tensor %33[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %35 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %37 = linalg.fill ins(%cst_0 : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %38 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %39 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %40 = loom.init_tensor %39[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %41 = linalg.fill ins(%cst_1 : f16) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %42 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %43 = loom.semaphore_take %42 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %44 = loom.init_tensor %43[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %45 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %46 = loom.semaphore_take %45 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %47 = loom.init_tensor %46[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %48 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %50 = loom.init_tensor %49[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %55 = loom.semaphore_take %54 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %56 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %57 = loom.semaphore_take %56 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %58 = loom.init_tensor %57[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %59 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %60 = loom.semaphore_take %59 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %61 = loom.init_tensor %60[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %62 = loom.semaphore_take %59 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %63 = loom.init_tensor %62[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %64 = loom.semaphore_take %59 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %65 = loom.init_tensor %64[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %66 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %67 = loom.semaphore_take %66 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %68 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%arg7, %c0_5, %25)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%68], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %69 = arith.muli %arg4, %c8 : index
              %70 = arith.addi %arg6, %69 : index
              loom.copy %reinterpret_cast_6, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %70], LR : [%arg5, %70]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %71 = loom.bufferize_to_tensor %55[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %72 = linalg.fill ins(%cst : f16) outs(%58 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %73 = linalg.batch_matmul ins(%24, %71 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%72 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %55 : memref<1x128x512xf16>
              loom.semaphore_give %22 : memref<1x32x128xf16>
              %74 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%73 : tensor<1x32x512xf16>) outs(%74 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %122 = arith.maximumf %in, %out : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%47 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.mulf %in_9, %cst_2 : f16
                %123 = arith.cmpf ogt, %in, %122 : f16
                %124 = arith.select %123, %in, %122 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %77 = loom.broadcast ins(%76 : tensor<1x32x1xf16>) outs(%65 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %77 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%58 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.mulf %in, %cst_2 : f16
                %123 = arith.subf %122, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %64 : memref<1x32x32xf16>
              %79 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x32x512xf16>) outs(%79 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %122 = arith.addf %in, %out : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.subf %in, %in_9 : f16
                %123 = math.exp %122 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %81, %80 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%37 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %122 = arith.mulf %in, %in_9 : f16
                %123 = arith.addf %122, %in_10 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              %83 = loom.broadcast ins(%81 : tensor<1x32x1xf16>) outs(%63 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%arg7, %25, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %70], LR : [%arg5, %70]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %85 = loom.bufferize_to_tensor %67[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %86 = linalg.fill ins(%cst : f16) outs(%44 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %87 = linalg.batch_matmul ins(%78, %85 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %67 : memref<1x512x128xf16>
              loom.semaphore_give %57 : memref<1x32x512xf16>
              %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %29, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%29 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %122 = arith.mulf %in_9, %in_10 : f16
                %123 = arith.addf %in, %122 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %62 : memref<1x32x32xf16>
              loom.semaphore_give %43 : memref<1x32x128xf16>
              %89 = linalg.copy ins(%76 : tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %46 : memref<1x32x1xf16>
              %90 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %91 = loom.semaphore_take %90 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %92 = loom.init_tensor %91[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %89 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%92 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = math.log %in : f16
                %123 = arith.addf %122, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %39 : memref<1x32x1xf16>
              %94 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%61 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %35 : memref<1x32x1xf16>
              %95 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %96 = loom.semaphore_take %95 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %97 = loom.init_tensor %96[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %94 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%97 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.divf %in, %in_9 : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %60 : memref<1x32x32xf16>
              loom.semaphore_give %27 : memref<1x32x128xf16>
              %99 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %100 = loom.semaphore_take %99 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %101 = loom.init_tensor %100[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %102 = arith.muli %arg4, %c8 : index
              %103 = arith.addi %arg6, %102 : index
              %104 = loom.gather ins(%93 : tensor<1x32x1xf16>) outs(%101 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %103], LR : [%c7, %103]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %91 : memref<1x32x1xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %107 = loom.init_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = loom.gather ins(%98 : tensor<1x32x128xf16>) outs(%107 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %103], LR : [%c7, %103]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %96 : memref<1x32x128xf16>
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %118 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
              %119 = loom.semaphore_take %118 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
              %120 = loom.init_tensor %119[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
              %121 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %121 {
                %122 = linalg.fill ins(%cst_1 : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%104 : tensor<16x1x32x1xf16>) outs(%122 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %134 = arith.maximumf %in, %out : f16
                  linalg.yield %134 : f16
                } -> tensor<1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%104, %123 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %134 = arith.subf %in, %in_12 : f16
                  %135 = math.exp %134 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %100 : memref<16x1x32x1xf16>
                loom.semaphore_give %33 : memref<1x32x1xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x1xf16>) outs(%125 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %134 = arith.addf %in, %out : f16
                  linalg.yield %134 : f16
                } -> tensor<1x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124, %126 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %134 = arith.divf %in, %in_12 : f16
                  linalg.yield %134 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %31 : memref<1x32x1xf16>
                %128 = loom.broadcast ins(%127 : tensor<16x1x32x1xf16>) outs(%120 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108, %128 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %134 = arith.mulf %in, %in_12 : f16
                  linalg.yield %134 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %119 : memref<16x1x32x32xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %130 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%129 : tensor<16x1x32x128xf16>) outs(%130 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %134 = arith.addf %in, %out : f16
                  linalg.yield %134 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %132 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%132], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %133 = loom.bufferize_to_memref %131 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                loom.copy %133, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %103], LR : [%arg5, %103]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %110 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x1x8_y8__d0i1_d1i1_d2i0__f01__dim_y_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (1) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c2 step %c1 {
              scf.for %arg8 = %c0 to %c2 step %c1 {
                %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg6, %arg8)
                %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %c0_3 = arith.constant 0 : index
                %c0_4 = arith.constant 0 : index
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_3, %c0_4)
                %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %25 = arith.addi %arg5, %arg4 : index
                loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%25, %c0], LR : [%25, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
                %26 = loom.bufferize_to_tensor %23[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %27 = arith.muli %21, %c512 : index
                %28 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %30 = loom.init_tensor %29[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %32 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %33 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %34 = loom.init_tensor %33[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %35 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %37 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %39 = linalg.fill ins(%cst_0 : f16) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %40 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %42 = loom.init_tensor %41[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %43 = linalg.fill ins(%cst_1 : f16) outs(%42 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %44 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %45 = loom.semaphore_take %44 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %46 = loom.init_tensor %45[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %47 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %48 = loom.semaphore_take %47 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %49 = loom.init_tensor %48[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %50 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %51 = loom.semaphore_take %50 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %52 = loom.init_tensor %51[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %53 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %54 = loom.semaphore_take %53 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %55 = loom.init_tensor %54[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %56 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
                %57 = loom.semaphore_take %56 : memref<1x128x512xf16> -> memref<1x128x512xf16>
                %58 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %59 = loom.semaphore_take %58 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %60 = loom.init_tensor %59[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %61 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
                %62 = loom.semaphore_take %61 : memref<1x32x32xf16> -> memref<1x32x32xf16>
                %63 = loom.init_tensor %62[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
                %64 = loom.semaphore_take %61 : memref<1x32x32xf16> -> memref<1x32x32xf16>
                %65 = loom.init_tensor %64[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
                %66 = loom.semaphore_take %61 : memref<1x32x32xf16> -> memref<1x32x32xf16>
                %67 = loom.init_tensor %66[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
                %68 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
                %69 = loom.semaphore_take %68 : memref<1x512x128xf16> -> memref<1x512x128xf16>
                %c0_5 = arith.constant 0 : index
                %70 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %27)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%70], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %reinterpret_cast_6, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
                %71 = loom.bufferize_to_tensor %57[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %72 = linalg.fill ins(%cst : f16) outs(%60 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                %73 = linalg.batch_matmul ins(%26, %71 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%72 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                loom.semaphore_give %57 : memref<1x128x512xf16>
                loom.semaphore_give %23 : memref<1x32x128xf16>
                %74 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%73 : tensor<1x32x512xf16>) outs(%74 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %120 = arith.maximumf %in, %out : f16
                  linalg.yield %120 : f16
                } -> tensor<1x32x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%49 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %120 = arith.mulf %in_9, %cst_2 : f16
                  %121 = arith.cmpf ogt, %in, %120 : f16
                  %122 = arith.select %121, %in, %120 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x1xf16>
                %77 = loom.broadcast ins(%76 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %77 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%60 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %120 = arith.mulf %in, %cst_2 : f16
                  %121 = arith.subf %120, %in_9 : f16
                  %122 = math.exp %121 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x512xf16>
                loom.semaphore_give %66 : memref<1x32x32xf16>
                %79 = linalg.fill ins(%cst : f16) outs(%52 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x32x512xf16>) outs(%79 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %120 = arith.addf %in, %out : f16
                  linalg.yield %120 : f16
                } -> tensor<1x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%55 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %120 = arith.subf %in, %in_9 : f16
                  %121 = math.exp %120 : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %81, %80 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%39 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %120 = arith.mulf %in, %in_9 : f16
                  %121 = arith.addf %120, %in_10 : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %51 : memref<1x32x1xf16>
                %83 = loom.broadcast ins(%81 : tensor<1x32x1xf16>) outs(%65 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %54 : memref<1x32x1xf16>
                %c0_7 = arith.constant 0 : index
                %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %27, %c0_7)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_8, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %85 = loom.bufferize_to_tensor %69[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %86 = linalg.fill ins(%cst : f16) outs(%46 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %87 = linalg.batch_matmul ins(%78, %85 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                loom.semaphore_give %69 : memref<1x512x128xf16>
                loom.semaphore_give %59 : memref<1x32x512xf16>
                %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %31, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%31 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %120 = arith.mulf %in_9, %in_10 : f16
                  %121 = arith.addf %in, %120 : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %64 : memref<1x32x32xf16>
                loom.semaphore_give %45 : memref<1x32x128xf16>
                %89 = linalg.copy ins(%76 : tensor<1x32x1xf16>) outs(%42 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                loom.semaphore_give %48 : memref<1x32x1xf16>
                %90 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %91 = loom.semaphore_take %90 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %92 = loom.init_tensor %91[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %89 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%92 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %120 = math.log %in : f16
                  %121 = arith.addf %120, %in_9 : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %41 : memref<1x32x1xf16>
                %94 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%63 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %37 : memref<1x32x1xf16>
                %95 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %96 = loom.semaphore_take %95 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %97 = loom.init_tensor %96[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %94 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%97 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %120 = arith.divf %in, %in_9 : f16
                  linalg.yield %120 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %62 : memref<1x32x32xf16>
                loom.semaphore_give %29 : memref<1x32x128xf16>
                %99 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %100 = loom.semaphore_take %99 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %101 = loom.init_tensor %100[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %102 = loom.gather ins(%93 : tensor<1x32x1xf16>) outs(%101 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%arg4, %arg6], LR : [%arg4, %arg6]) -> tensor<16x1x32x1xf16>
                loom.semaphore_give %91 : memref<1x32x1xf16>
                %103 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %104 = loom.semaphore_take %103 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %105 = loom.init_tensor %104[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %106 = loom.gather ins(%98 : tensor<1x32x128xf16>) outs(%105 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%arg4, %arg6], LR : [%arg4, %arg6]) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %96 : memref<1x32x128xf16>
                %107 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %108 = loom.semaphore_take %107 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %109 = loom.init_tensor %108[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %110 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %111 = loom.semaphore_take %110 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %112 = loom.init_tensor %111[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %113 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %114 = loom.semaphore_take %113 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %115 = loom.init_tensor %114[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %116 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
                %117 = loom.semaphore_take %116 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
                %118 = loom.init_tensor %117[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
                %119 = arith.cmpi eq, %arg5, %c0 : index
                scf.if %119 {
                  %120 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%102 : tensor<16x1x32x1xf16>) outs(%120 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.maximumf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<1x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %121 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%112 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %132 = arith.subf %in, %in_12 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %100 : memref<16x1x32x1xf16>
                  loom.semaphore_give %35 : memref<1x32x1xf16>
                  %123 = linalg.fill ins(%cst : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%122 : tensor<16x1x32x1xf16>) outs(%123 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.addf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<1x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%122, %124 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%112 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %132 = arith.divf %in, %in_12 : f16
                    linalg.yield %132 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %33 : memref<1x32x1xf16>
                  %126 = loom.broadcast ins(%125 : tensor<16x1x32x1xf16>) outs(%118 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %111 : memref<16x1x32x1xf16>
                  %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %126 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%115 : tensor<16x1x32x128xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %132 = arith.mulf %in, %in_12 : f16
                    linalg.yield %132 : f16
                  } -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %117 : memref<16x1x32x32xf16>
                  loom.semaphore_give %104 : memref<16x1x32x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%109 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                  %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%127 : tensor<16x1x32x128xf16>) outs(%128 : tensor<1x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.addf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<1x32x128xf16>
                  loom.semaphore_give %114 : memref<16x1x32x128xf16>
                  %c0_9 = arith.constant 0 : index
                  %c0_10 = arith.constant 0 : index
                  %130 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%130], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %131 = loom.bufferize_to_memref %129 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                  loom.copy %131, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %108 : memref<1x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x2x4_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc2_dim_y_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (2) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg5, %arg6)
              %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %25 = arith.muli %arg4, %c2 : index
              %26 = arith.addi %25, %c1 : index
              loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 8] region : (UL : [%25, %c0], LR : [%26, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %27 = loom.bufferize_to_tensor %23[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %28 = arith.muli %21, %c512 : index
              %29 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %30 = loom.semaphore_take %29 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %31 = loom.init_tensor %30[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %35 = loom.init_tensor %34[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %36 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %37 = loom.init_tensor %36[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %38 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %39 = loom.init_tensor %38[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %40 = linalg.fill ins(%cst_0 : f16) outs(%39 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %43 = loom.init_tensor %42[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %44 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %45 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %46 = loom.semaphore_take %45 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %47 = loom.init_tensor %46[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %48 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %50 = loom.init_tensor %49[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %56 = loom.init_tensor %55[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %57 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %58 = loom.semaphore_take %57 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %59 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %60 = loom.semaphore_take %59 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %61 = loom.init_tensor %60[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %62 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %64 = loom.init_tensor %63[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %65 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %66 = loom.init_tensor %65[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %67 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %68 = loom.init_tensor %67[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %69 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %70 = loom.semaphore_take %69 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %28)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %72 = arith.addi %arg5, %25 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%72, %arg6], LR : [%72, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %73 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %74 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %75 = linalg.batch_matmul ins(%27, %73 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%74 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              loom.semaphore_give %23 : memref<1x32x128xf16>
              %76 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%75 : tensor<1x32x512xf16>) outs(%76 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %122 = arith.maximumf %in, %out : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.mulf %in_9, %cst_2 : f16
                %123 = arith.cmpf ogt, %in, %122 : f16
                %124 = arith.select %123, %in, %122 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %79 = loom.broadcast ins(%78 : tensor<1x32x1xf16>) outs(%68 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %79 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.mulf %in, %cst_2 : f16
                %123 = arith.subf %122, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %67 : memref<1x32x32xf16>
              %81 = linalg.fill ins(%cst : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%80 : tensor<1x32x512xf16>) outs(%81 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %122 = arith.addf %in, %out : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.subf %in, %in_9 : f16
                %123 = math.exp %122 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %83, %82 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %122 = arith.mulf %in, %in_9 : f16
                %123 = arith.addf %122, %in_10 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %85 = loom.broadcast ins(%83 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %28, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%72, %arg6], LR : [%72, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %87 = loom.bufferize_to_tensor %70[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %88 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %89 = linalg.batch_matmul ins(%80, %87 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%88 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %70 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %32, %85 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%32 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %122 = arith.mulf %in_9, %in_10 : f16
                %123 = arith.addf %in, %122 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %65 : memref<1x32x32xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %91 = linalg.copy ins(%78 : tensor<1x32x1xf16>) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              %92 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %93 = loom.semaphore_take %92 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %94 = loom.init_tensor %93[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %91 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%94 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = math.log %in : f16
                %123 = arith.addf %122, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %42 : memref<1x32x1xf16>
              %96 = loom.broadcast ins(%84 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %38 : memref<1x32x1xf16>
              %97 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %98 = loom.semaphore_take %97 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %99 = loom.init_tensor %98[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %96 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%99 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.divf %in, %in_9 : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %63 : memref<1x32x32xf16>
              loom.semaphore_give %30 : memref<1x32x128xf16>
              %101 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %102 = loom.semaphore_take %101 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %103 = loom.init_tensor %102[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = loom.gather ins(%95 : tensor<1x32x1xf16>) outs(%103 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%25, %arg6], LR : [%26, %arg6]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %93 : memref<1x32x1xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %107 = loom.init_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = loom.gather ins(%100 : tensor<1x32x128xf16>) outs(%107 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%25, %arg6], LR : [%26, %arg6]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %98 : memref<1x32x128xf16>
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %118 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
              %119 = loom.semaphore_take %118 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
              %120 = loom.init_tensor %119[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
              %121 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %121 {
                %122 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%104 : tensor<16x1x32x1xf16>) outs(%122 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.maximumf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%104, %123 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.subf %in, %in_12 : f16
                  %136 = math.exp %135 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %102 : memref<16x1x32x1xf16>
                loom.semaphore_give %36 : memref<1x32x1xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x1xf16>) outs(%125 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124, %126 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.divf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %128 = loom.broadcast ins(%127 : tensor<16x1x32x1xf16>) outs(%120 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108, %128 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.mulf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %119 : memref<16x1x32x32xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %130 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%129 : tensor<16x1x32x128xf16>) outs(%130 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %132 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%132], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %133 = loom.bufferize_to_memref %131 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %134 = arith.addi %arg5, %25 : index
                loom.copy %133, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%134, %arg6], LR : [%134, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %110 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x4x2_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc4_dim_y_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (4) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%arg5, %arg6)
              %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %25 = arith.muli %arg4, %c4 : index
              %26 = arith.addi %25, %c3 : index
              loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%25, %c0], LR : [%26, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %27 = loom.bufferize_to_tensor %23[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %28 = arith.muli %21, %c512 : index
              %29 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %30 = loom.semaphore_take %29 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %31 = loom.init_tensor %30[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %33 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %34 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %35 = loom.init_tensor %34[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %36 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %37 = loom.init_tensor %36[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %38 = loom.semaphore_take %33 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %39 = loom.init_tensor %38[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %40 = linalg.fill ins(%cst_0 : f16) outs(%39 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %43 = loom.init_tensor %42[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %44 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %45 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %46 = loom.semaphore_take %45 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %47 = loom.init_tensor %46[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %48 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %50 = loom.init_tensor %49[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %56 = loom.init_tensor %55[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %57 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %58 = loom.semaphore_take %57 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %59 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %60 = loom.semaphore_take %59 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %61 = loom.init_tensor %60[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %62 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %64 = loom.init_tensor %63[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %65 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %66 = loom.init_tensor %65[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %67 = loom.semaphore_take %62 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %68 = loom.init_tensor %67[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %69 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %70 = loom.semaphore_take %69 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %28)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %72 = arith.addi %arg5, %25 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%72, %arg6], LR : [%72, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %73 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %74 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %75 = linalg.batch_matmul ins(%27, %73 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%74 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              loom.semaphore_give %23 : memref<1x32x128xf16>
              %76 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%75 : tensor<1x32x512xf16>) outs(%76 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %122 = arith.maximumf %in, %out : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.mulf %in_9, %cst_2 : f16
                %123 = arith.cmpf ogt, %in, %122 : f16
                %124 = arith.select %123, %in, %122 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %79 = loom.broadcast ins(%78 : tensor<1x32x1xf16>) outs(%68 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75, %79 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.mulf %in, %cst_2 : f16
                %123 = arith.subf %122, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %67 : memref<1x32x32xf16>
              %81 = linalg.fill ins(%cst : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%80 : tensor<1x32x512xf16>) outs(%81 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %122 = arith.addf %in, %out : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.subf %in, %in_9 : f16
                %123 = math.exp %122 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %83, %82 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %122 = arith.mulf %in, %in_9 : f16
                %123 = arith.addf %122, %in_10 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %85 = loom.broadcast ins(%83 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %28, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%72, %arg6], LR : [%72, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %87 = loom.bufferize_to_tensor %70[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %88 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %89 = linalg.batch_matmul ins(%80, %87 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%88 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %70 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %32, %85 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%32 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %122 = arith.mulf %in_9, %in_10 : f16
                %123 = arith.addf %in, %122 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %65 : memref<1x32x32xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %91 = linalg.copy ins(%78 : tensor<1x32x1xf16>) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              %92 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %93 = loom.semaphore_take %92 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %94 = loom.init_tensor %93[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %91 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%94 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = math.log %in : f16
                %123 = arith.addf %122, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %42 : memref<1x32x1xf16>
              %96 = loom.broadcast ins(%84 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %38 : memref<1x32x1xf16>
              %97 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %98 = loom.semaphore_take %97 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %99 = loom.init_tensor %98[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %96 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%99 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.divf %in, %in_9 : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %63 : memref<1x32x32xf16>
              loom.semaphore_give %30 : memref<1x32x128xf16>
              %101 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %102 = loom.semaphore_take %101 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %103 = loom.init_tensor %102[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = loom.gather ins(%95 : tensor<1x32x1xf16>) outs(%103 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%25, %arg6], LR : [%26, %arg6]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %93 : memref<1x32x1xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %107 = loom.init_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = loom.gather ins(%100 : tensor<1x32x128xf16>) outs(%107 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%25, %arg6], LR : [%26, %arg6]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %98 : memref<1x32x128xf16>
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %118 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
              %119 = loom.semaphore_take %118 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
              %120 = loom.init_tensor %119[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
              %121 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %121 {
                %122 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%104 : tensor<16x1x32x1xf16>) outs(%122 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.maximumf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%104, %123 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.subf %in, %in_12 : f16
                  %136 = math.exp %135 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %102 : memref<16x1x32x1xf16>
                loom.semaphore_give %36 : memref<1x32x1xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x1xf16>) outs(%125 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124, %126 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.divf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %128 = loom.broadcast ins(%127 : tensor<16x1x32x1xf16>) outs(%120 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108, %128 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.mulf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %119 : memref<16x1x32x32xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %130 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%129 : tensor<16x1x32x128xf16>) outs(%130 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %132 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%132], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %133 = loom.bufferize_to_memref %131 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %134 = arith.addi %arg5, %25 : index
                loom.copy %133, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%134, %arg6], LR : [%134, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %110 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8x1_y8__d0i1_d1i1_d2i0__f01__dim_x_level1_bc8_dim_y_level0_bc8_n_n_n__tile_b1__tile_n512__tile_s512(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c16 = arith.constant 16 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c16 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %21 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %22 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %24 = loom.bufferize_to_tensor %22[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %25 = arith.muli %20, %c512 : index
              %26 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %27 = loom.semaphore_take %26 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %28 = loom.init_tensor %27[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %30 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %31 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %32 = loom.init_tensor %31[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %33 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %34 = loom.init_tensor %33[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %35 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %37 = linalg.fill ins(%cst_0 : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %38 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %39 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %40 = loom.init_tensor %39[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %41 = linalg.fill ins(%cst_1 : f16) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %42 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %43 = loom.semaphore_take %42 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %44 = loom.init_tensor %43[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %45 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %46 = loom.semaphore_take %45 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %47 = loom.init_tensor %46[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %48 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %50 = loom.init_tensor %49[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %55 = loom.semaphore_take %54 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %56 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %57 = loom.semaphore_take %56 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %58 = loom.init_tensor %57[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %59 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %60 = loom.semaphore_take %59 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %61 = loom.init_tensor %60[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %62 = loom.semaphore_take %59 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %63 = loom.init_tensor %62[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %64 = loom.semaphore_take %59 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %65 = loom.init_tensor %64[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %66 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %67 = loom.semaphore_take %66 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %68 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%arg7, %c0_5, %25)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%68], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %69 = arith.muli %arg4, %c8 : index
              %70 = arith.addi %arg5, %69 : index
              loom.copy %reinterpret_cast_6, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%70, %arg6], LR : [%70, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %71 = loom.bufferize_to_tensor %55[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %72 = linalg.fill ins(%cst : f16) outs(%58 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %73 = linalg.batch_matmul ins(%24, %71 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%72 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %55 : memref<1x128x512xf16>
              loom.semaphore_give %22 : memref<1x32x128xf16>
              %74 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%73 : tensor<1x32x512xf16>) outs(%74 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %122 = arith.maximumf %in, %out : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%47 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.mulf %in_9, %cst_2 : f16
                %123 = arith.cmpf ogt, %in, %122 : f16
                %124 = arith.select %123, %in, %122 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %77 = loom.broadcast ins(%76 : tensor<1x32x1xf16>) outs(%65 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %77 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%58 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.mulf %in, %cst_2 : f16
                %123 = arith.subf %122, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %64 : memref<1x32x32xf16>
              %79 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x32x512xf16>) outs(%79 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %122 = arith.addf %in, %out : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.subf %in, %in_9 : f16
                %123 = math.exp %122 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %81, %80 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%37 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %122 = arith.mulf %in, %in_9 : f16
                %123 = arith.addf %122, %in_10 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              %83 = loom.broadcast ins(%81 : tensor<1x32x1xf16>) outs(%63 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%arg7, %25, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%70, %arg6], LR : [%70, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %85 = loom.bufferize_to_tensor %67[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %86 = linalg.fill ins(%cst : f16) outs(%44 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %87 = linalg.batch_matmul ins(%78, %85 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %67 : memref<1x512x128xf16>
              loom.semaphore_give %57 : memref<1x32x512xf16>
              %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %29, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%29 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %122 = arith.mulf %in_9, %in_10 : f16
                %123 = arith.addf %in, %122 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %62 : memref<1x32x32xf16>
              loom.semaphore_give %43 : memref<1x32x128xf16>
              %89 = linalg.copy ins(%76 : tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %46 : memref<1x32x1xf16>
              %90 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %91 = loom.semaphore_take %90 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %92 = loom.init_tensor %91[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %89 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%92 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = math.log %in : f16
                %123 = arith.addf %122, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %39 : memref<1x32x1xf16>
              %94 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%61 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %35 : memref<1x32x1xf16>
              %95 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %96 = loom.semaphore_take %95 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %97 = loom.init_tensor %96[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %94 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%97 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %122 = arith.divf %in, %in_9 : f16
                linalg.yield %122 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %60 : memref<1x32x32xf16>
              loom.semaphore_give %27 : memref<1x32x128xf16>
              %99 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %100 = loom.semaphore_take %99 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %101 = loom.init_tensor %100[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %102 = arith.muli %arg4, %c8 : index
              %103 = arith.addi %102, %c7 : index
              %104 = loom.gather ins(%93 : tensor<1x32x1xf16>) outs(%101 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%102, %arg6], LR : [%103, %arg6]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %91 : memref<1x32x1xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %107 = loom.init_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = loom.gather ins(%98 : tensor<1x32x128xf16>) outs(%107 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%102, %arg6], LR : [%103, %arg6]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %96 : memref<1x32x128xf16>
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %118 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
              %119 = loom.semaphore_take %118 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
              %120 = loom.init_tensor %119[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
              %121 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %121 {
                %122 = linalg.fill ins(%cst_1 : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%104 : tensor<16x1x32x1xf16>) outs(%122 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.maximumf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%104, %123 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.subf %in, %in_12 : f16
                  %136 = math.exp %135 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %100 : memref<16x1x32x1xf16>
                loom.semaphore_give %33 : memref<1x32x1xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x1xf16>) outs(%125 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%124, %126 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.divf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %31 : memref<1x32x1xf16>
                %128 = loom.broadcast ins(%127 : tensor<16x1x32x1xf16>) outs(%120 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108, %128 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %135 = arith.mulf %in, %in_12 : f16
                  linalg.yield %135 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %119 : memref<16x1x32x32xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %130 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%129 : tensor<16x1x32x128xf16>) outs(%130 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %135 = arith.addf %in, %out : f16
                  linalg.yield %135 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %132 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%132], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %133 = loom.bufferize_to_memref %131 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %134 = arith.addi %arg5, %102 : index
                loom.copy %133, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%134, %arg6], LR : [%134, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %110 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
