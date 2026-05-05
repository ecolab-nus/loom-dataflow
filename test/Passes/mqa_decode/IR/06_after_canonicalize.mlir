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
                %30 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
                %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %27)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
                %72 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %73 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                %74 = linalg.batch_matmul ins(%26, %72 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                loom.semaphore_give %58 : memref<1x128x512xf16>
                loom.semaphore_give %23 : memref<1x32x128xf16>
                %75 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x32x512xf16>) outs(%75 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %122 = arith.maximumf %in, %out : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x1xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %122 = arith.mulf %in_9, %cst_2 : f16
                  %123 = arith.cmpf ogt, %in, %122 : f16
                  %124 = arith.select %123, %in, %122 : f16
                  linalg.yield %124 : f16
                } -> tensor<1x32x1xf16>
                %78 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%68 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %122 = arith.mulf %in, %cst_2 : f16
                  %123 = arith.subf %122, %in_9 : f16
                  %124 = math.exp %123 : f16
                  linalg.yield %124 : f16
                } -> tensor<1x32x512xf16>
                loom.semaphore_give %67 : memref<1x32x32xf16>
                %80 = linalg.fill ins(%cst : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x32x512xf16>) outs(%80 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %122 = arith.addf %in, %out : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %122 = arith.subf %in, %in_9 : f16
                  %123 = math.exp %122 : f16
                  linalg.yield %123 : f16
                } -> tensor<1x32x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %82, %81 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %122 = arith.mulf %in, %in_9 : f16
                  %123 = arith.addf %122, %in_10 : f16
                  linalg.yield %123 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %52 : memref<1x32x1xf16>
                %84 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %55 : memref<1x32x1xf16>
                %c0_7 = arith.constant 0 : index
                %85 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %27, %c0_7)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%85], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_8, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %86 = loom.bufferize_to_tensor %70[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %87 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %88 = linalg.batch_matmul ins(%79, %86 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%87 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                loom.semaphore_give %70 : memref<1x512x128xf16>
                loom.semaphore_give %60 : memref<1x32x512xf16>
                %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %32, %84 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%32 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %122 = arith.mulf %in_9, %in_10 : f16
                  %123 = arith.addf %in, %122 : f16
                  linalg.yield %123 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %65 : memref<1x32x32xf16>
                loom.semaphore_give %46 : memref<1x32x128xf16>
                %90 = linalg.copy ins(%77 : tensor<1x32x1xf16>) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                loom.semaphore_give %49 : memref<1x32x1xf16>
                %91 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %92 = loom.semaphore_take %91 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %93 = loom.init_tensor %92[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83, %90 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%93 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %122 = math.log %in : f16
                  %123 = arith.addf %122, %in_9 : f16
                  linalg.yield %123 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %42 : memref<1x32x1xf16>
                %95 = loom.broadcast ins(%83 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %38 : memref<1x32x1xf16>
                %96 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %98 = loom.init_tensor %97[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %95 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%98 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %122 = arith.divf %in, %in_9 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %63 : memref<1x32x32xf16>
                loom.semaphore_give %30 : memref<1x32x128xf16>
                %100 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %101 = loom.semaphore_take %100 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %102 = loom.init_tensor %101[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %103 = arith.addi %arg6, %arg4 : index
                %104 = loom.gather ins(%94 : tensor<1x32x1xf16>) outs(%102 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %103], LR : [%c7, %103]) -> tensor<16x1x32x1xf16>
                loom.semaphore_give %92 : memref<1x32x1xf16>
                %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %107 = loom.init_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %108 = loom.gather ins(%99 : tensor<1x32x128xf16>) outs(%107 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %103], LR : [%c7, %103]) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %97 : memref<1x32x128xf16>
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
                    %134 = arith.maximumf %in, %out : f16
                    linalg.yield %134 : f16
                  } -> tensor<1x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%104, %123 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %134 = arith.subf %in, %in_12 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %101 : memref<16x1x32x1xf16>
                  loom.semaphore_give %36 : memref<1x32x1xf16>
                  %125 = linalg.fill ins(%cst : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
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
                  loom.semaphore_give %34 : memref<1x32x1xf16>
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
                  loom.semaphore_give %29 : memref<1x32x128xf16>
                  %c0_9 = arith.constant 0 : index
                  %c0_10 = arith.constant 0 : index
                  %132 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%132], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %133 = loom.bufferize_to_memref %131 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                  loom.copy %133, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %110 : memref<1x32x128xf16>
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
              %31 = loom.semaphore_take %29 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %32 = loom.init_tensor %31[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %33 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %34 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %35 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %37 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %39 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %40 = loom.init_tensor %39[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %44 = loom.init_tensor %43[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %46 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %47 = loom.semaphore_take %46 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %48 = loom.init_tensor %47[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %49 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %51 = loom.init_tensor %50[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %52 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %53 = loom.semaphore_take %52 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %54 = loom.init_tensor %53[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %55 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %56 = loom.semaphore_take %55 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %57 = loom.init_tensor %56[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %58 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %59 = loom.semaphore_take %58 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %60 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %61 = loom.semaphore_take %60 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %62 = loom.init_tensor %61[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %63 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %64 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %65 = loom.init_tensor %64[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %66 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %67 = loom.init_tensor %66[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %68 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %69 = loom.init_tensor %68[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %70 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %71 = loom.semaphore_take %70 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %28)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %73 = arith.addi %arg6, %25 : index
              loom.copy %reinterpret_cast_6, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %73], LR : [%arg5, %73]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %74 = loom.bufferize_to_tensor %59[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %75 = linalg.fill ins(%cst : f16) outs(%62 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %76 = linalg.batch_matmul ins(%27, %74 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%75 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %59 : memref<1x128x512xf16>
              loom.semaphore_give %23 : memref<1x32x128xf16>
              %77 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x32x512xf16>) outs(%77 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %124 = arith.maximumf %in, %out : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%51 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.mulf %in_9, %cst_2 : f16
                %125 = arith.cmpf ogt, %in, %124 : f16
                %126 = arith.select %125, %in, %124 : f16
                linalg.yield %126 : f16
              } -> tensor<1x32x1xf16>
              %80 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%69 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %80 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%62 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.mulf %in, %cst_2 : f16
                %125 = arith.subf %124, %in_9 : f16
                %126 = math.exp %125 : f16
                linalg.yield %126 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %68 : memref<1x32x32xf16>
              %82 = linalg.fill ins(%cst : f16) outs(%54 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<1x32x512xf16>) outs(%82 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %124 = arith.addf %in, %out : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %79 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%57 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.subf %in, %in_9 : f16
                %125 = math.exp %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %84, %83 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %124 = arith.mulf %in, %in_9 : f16
                %125 = arith.addf %124, %in_10 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %53 : memref<1x32x1xf16>
              %86 = loom.broadcast ins(%84 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %56 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %87 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %28, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%87], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %73], LR : [%arg5, %73]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %88 = loom.bufferize_to_tensor %71[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %89 = linalg.fill ins(%cst : f16) outs(%48 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %90 = linalg.batch_matmul ins(%81, %88 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%89 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %71 : memref<1x512x128xf16>
              loom.semaphore_give %61 : memref<1x32x512xf16>
              %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %33, %86 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%33 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %124 = arith.mulf %in_9, %in_10 : f16
                %125 = arith.addf %in, %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x32x32xf16>
              loom.semaphore_give %47 : memref<1x32x128xf16>
              %92 = linalg.copy ins(%79 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %50 : memref<1x32x1xf16>
              %93 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %94 = loom.semaphore_take %93 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %95 = loom.init_tensor %94[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %92 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%95 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = math.log %in : f16
                %125 = arith.addf %124, %in_9 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %43 : memref<1x32x1xf16>
              %97 = loom.broadcast ins(%85 : tensor<1x32x1xf16>) outs(%65 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %39 : memref<1x32x1xf16>
              %98 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %99 = loom.semaphore_take %98 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %100 = loom.init_tensor %99[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %97 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%100 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.divf %in, %in_9 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %64 : memref<1x32x32xf16>
              loom.semaphore_give %31 : memref<1x32x128xf16>
              %102 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %103 = loom.semaphore_take %102 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %104 = loom.init_tensor %103[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %105 = arith.addi %arg6, %25 : index
              %106 = loom.gather ins(%96 : tensor<1x32x1xf16>) outs(%104 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %105], LR : [%c7, %105]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %94 : memref<1x32x1xf16>
              %107 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %108 = loom.semaphore_take %107 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %109 = loom.init_tensor %108[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %110 = loom.gather ins(%101 : tensor<1x32x128xf16>) outs(%109 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %105], LR : [%c7, %105]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %99 : memref<1x32x128xf16>
              %111 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %112 = loom.semaphore_take %111 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %113 = loom.init_tensor %112[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %114 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %115 = loom.semaphore_take %114 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %116 = loom.init_tensor %115[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %117 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %118 = loom.semaphore_take %117 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %119 = loom.init_tensor %118[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %120 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
              %121 = loom.semaphore_take %120 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
              %122 = loom.init_tensor %121[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
              %123 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %123 {
                %124 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%106 : tensor<16x1x32x1xf16>) outs(%124 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.maximumf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %125 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%116 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.subf %in, %in_12 : f16
                  %137 = math.exp %136 : f16
                  linalg.yield %137 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %103 : memref<16x1x32x1xf16>
                loom.semaphore_give %37 : memref<1x32x1xf16>
                %127 = linalg.fill ins(%cst : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%126 : tensor<16x1x32x1xf16>) outs(%127 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126, %128 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%116 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.divf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %35 : memref<1x32x1xf16>
                %130 = loom.broadcast ins(%129 : tensor<16x1x32x1xf16>) outs(%122 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %115 : memref<16x1x32x1xf16>
                %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %130 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%119 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.mulf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %121 : memref<16x1x32x32xf16>
                loom.semaphore_give %108 : memref<16x1x32x128xf16>
                %132 = linalg.fill ins(%cst : f16) outs(%113 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %133 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%131 : tensor<16x1x32x128xf16>) outs(%132 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %118 : memref<16x1x32x128xf16>
                loom.semaphore_give %30 : memref<1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %134 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%134], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %135 = loom.bufferize_to_memref %133 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                loom.copy %135, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %105], LR : [%arg5, %105]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %112 : memref<1x32x128xf16>
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
              %31 = loom.semaphore_take %29 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %32 = loom.init_tensor %31[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %33 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %34 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %35 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %37 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %39 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %40 = loom.init_tensor %39[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %44 = loom.init_tensor %43[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %46 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %47 = loom.semaphore_take %46 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %48 = loom.init_tensor %47[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %49 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %51 = loom.init_tensor %50[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %52 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %53 = loom.semaphore_take %52 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %54 = loom.init_tensor %53[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %55 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %56 = loom.semaphore_take %55 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %57 = loom.init_tensor %56[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %58 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %59 = loom.semaphore_take %58 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %60 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %61 = loom.semaphore_take %60 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %62 = loom.init_tensor %61[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %63 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %64 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %65 = loom.init_tensor %64[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %66 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %67 = loom.init_tensor %66[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %68 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %69 = loom.init_tensor %68[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %70 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %71 = loom.semaphore_take %70 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %28)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %73 = arith.addi %arg6, %25 : index
              loom.copy %reinterpret_cast_6, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %73], LR : [%arg5, %73]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %74 = loom.bufferize_to_tensor %59[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %75 = linalg.fill ins(%cst : f16) outs(%62 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %76 = linalg.batch_matmul ins(%27, %74 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%75 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %59 : memref<1x128x512xf16>
              loom.semaphore_give %23 : memref<1x32x128xf16>
              %77 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x32x512xf16>) outs(%77 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %124 = arith.maximumf %in, %out : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%51 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.mulf %in_9, %cst_2 : f16
                %125 = arith.cmpf ogt, %in, %124 : f16
                %126 = arith.select %125, %in, %124 : f16
                linalg.yield %126 : f16
              } -> tensor<1x32x1xf16>
              %80 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%69 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %80 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%62 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.mulf %in, %cst_2 : f16
                %125 = arith.subf %124, %in_9 : f16
                %126 = math.exp %125 : f16
                linalg.yield %126 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %68 : memref<1x32x32xf16>
              %82 = linalg.fill ins(%cst : f16) outs(%54 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<1x32x512xf16>) outs(%82 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %124 = arith.addf %in, %out : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %79 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%57 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.subf %in, %in_9 : f16
                %125 = math.exp %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %84, %83 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %124 = arith.mulf %in, %in_9 : f16
                %125 = arith.addf %124, %in_10 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %53 : memref<1x32x1xf16>
              %86 = loom.broadcast ins(%84 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %56 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %87 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %28, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%87], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %73], LR : [%arg5, %73]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %88 = loom.bufferize_to_tensor %71[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %89 = linalg.fill ins(%cst : f16) outs(%48 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %90 = linalg.batch_matmul ins(%81, %88 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%89 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %71 : memref<1x512x128xf16>
              loom.semaphore_give %61 : memref<1x32x512xf16>
              %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %33, %86 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%33 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %124 = arith.mulf %in_9, %in_10 : f16
                %125 = arith.addf %in, %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x32x32xf16>
              loom.semaphore_give %47 : memref<1x32x128xf16>
              %92 = linalg.copy ins(%79 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %50 : memref<1x32x1xf16>
              %93 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %94 = loom.semaphore_take %93 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %95 = loom.init_tensor %94[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %92 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%95 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = math.log %in : f16
                %125 = arith.addf %124, %in_9 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %43 : memref<1x32x1xf16>
              %97 = loom.broadcast ins(%85 : tensor<1x32x1xf16>) outs(%65 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %39 : memref<1x32x1xf16>
              %98 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %99 = loom.semaphore_take %98 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %100 = loom.init_tensor %99[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %97 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%100 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.divf %in, %in_9 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %64 : memref<1x32x32xf16>
              loom.semaphore_give %31 : memref<1x32x128xf16>
              %102 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %103 = loom.semaphore_take %102 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %104 = loom.init_tensor %103[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %105 = arith.addi %arg6, %25 : index
              %106 = loom.gather ins(%96 : tensor<1x32x1xf16>) outs(%104 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %105], LR : [%c7, %105]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %94 : memref<1x32x1xf16>
              %107 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %108 = loom.semaphore_take %107 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %109 = loom.init_tensor %108[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %110 = loom.gather ins(%101 : tensor<1x32x128xf16>) outs(%109 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %105], LR : [%c7, %105]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %99 : memref<1x32x128xf16>
              %111 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %112 = loom.semaphore_take %111 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %113 = loom.init_tensor %112[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %114 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %115 = loom.semaphore_take %114 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %116 = loom.init_tensor %115[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %117 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %118 = loom.semaphore_take %117 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %119 = loom.init_tensor %118[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %120 = loom.alloc [16, 1, 32, 32] on @L1 : memref<16x1x32x32xf16>
              %121 = loom.semaphore_take %120 : memref<16x1x32x32xf16> -> memref<16x1x32x32xf16>
              %122 = loom.init_tensor %121[16, 1, 32, 32] : memref<16x1x32x32xf16> -> tensor<16x1x32x32xf16>
              %123 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %123 {
                %124 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%106 : tensor<16x1x32x1xf16>) outs(%124 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.maximumf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %125 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%116 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.subf %in, %in_12 : f16
                  %137 = math.exp %136 : f16
                  linalg.yield %137 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %103 : memref<16x1x32x1xf16>
                loom.semaphore_give %37 : memref<1x32x1xf16>
                %127 = linalg.fill ins(%cst : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%126 : tensor<16x1x32x1xf16>) outs(%127 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%126, %128 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%116 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.divf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %35 : memref<1x32x1xf16>
                %130 = loom.broadcast ins(%129 : tensor<16x1x32x1xf16>) outs(%122 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %115 : memref<16x1x32x1xf16>
                %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %130 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%119 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.mulf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %121 : memref<16x1x32x32xf16>
                loom.semaphore_give %108 : memref<16x1x32x128xf16>
                %132 = linalg.fill ins(%cst : f16) outs(%113 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %133 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%131 : tensor<16x1x32x128xf16>) outs(%132 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %118 : memref<16x1x32x128xf16>
                loom.semaphore_give %30 : memref<1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %134 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%134], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %135 = loom.bufferize_to_memref %133 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                loom.copy %135, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %105], LR : [%arg5, %105]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %112 : memref<1x32x128xf16>
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
              %28 = loom.semaphore_take %26 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %29 = loom.init_tensor %28[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %31 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %32 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %33 = loom.init_tensor %32[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %34 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %35 = loom.init_tensor %34[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %36 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %37 = loom.init_tensor %36[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %38 = linalg.fill ins(%cst_0 : f16) outs(%37 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %39 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %40 = loom.semaphore_take %39 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %41 = loom.init_tensor %40[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %42 = linalg.fill ins(%cst_1 : f16) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %43 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %44 = loom.semaphore_take %43 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %45 = loom.init_tensor %44[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %46 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %47 = loom.semaphore_take %46 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %48 = loom.init_tensor %47[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %49 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %51 = loom.init_tensor %50[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %52 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %53 = loom.semaphore_take %52 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %54 = loom.init_tensor %53[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %55 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %56 = loom.semaphore_take %55 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %57 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %58 = loom.semaphore_take %57 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %59 = loom.init_tensor %58[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %60 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %61 = loom.semaphore_take %60 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %62 = loom.init_tensor %61[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %63 = loom.semaphore_take %60 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %64 = loom.init_tensor %63[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %65 = loom.semaphore_take %60 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %66 = loom.init_tensor %65[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %67 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %68 = loom.semaphore_take %67 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%arg7, %c0_5, %25)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%69], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %70 = arith.muli %arg4, %c8 : index
              %71 = arith.addi %arg6, %70 : index
              loom.copy %reinterpret_cast_6, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %71], LR : [%arg5, %71]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %72 = loom.bufferize_to_tensor %56[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %73 = linalg.fill ins(%cst : f16) outs(%59 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %74 = linalg.batch_matmul ins(%24, %72 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %56 : memref<1x128x512xf16>
              loom.semaphore_give %22 : memref<1x32x128xf16>
              %75 = linalg.fill ins(%cst_1 : f16) outs(%48 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x32x512xf16>) outs(%75 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.maximumf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%48 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in_9, %cst_2 : f16
                %124 = arith.cmpf ogt, %in, %123 : f16
                %125 = arith.select %124, %in, %123 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              %78 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%59 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in, %cst_2 : f16
                %124 = arith.subf %123, %in_9 : f16
                %125 = math.exp %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %65 : memref<1x32x32xf16>
              %80 = linalg.fill ins(%cst : f16) outs(%51 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x32x512xf16>) outs(%80 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.addf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%54 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.subf %in, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %82, %81 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in, %in_9 : f16
                %124 = arith.addf %123, %in_10 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %50 : memref<1x32x1xf16>
              %84 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %53 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %85 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%arg7, %25, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%85], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %71], LR : [%arg5, %71]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %86 = loom.bufferize_to_tensor %68[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %87 = linalg.fill ins(%cst : f16) outs(%45 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %88 = linalg.batch_matmul ins(%79, %86 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%87 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %68 : memref<1x512x128xf16>
              loom.semaphore_give %58 : memref<1x32x512xf16>
              %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %30, %84 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in_9, %in_10 : f16
                %124 = arith.addf %in, %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %63 : memref<1x32x32xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              %90 = linalg.copy ins(%77 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %47 : memref<1x32x1xf16>
              %91 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %92 = loom.semaphore_take %91 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %93 = loom.init_tensor %92[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83, %90 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%93 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = math.log %in : f16
                %124 = arith.addf %123, %in_9 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %95 = loom.broadcast ins(%83 : tensor<1x32x1xf16>) outs(%62 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %96 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %97 = loom.semaphore_take %96 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %98 = loom.init_tensor %97[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %95 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%98 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.divf %in, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %61 : memref<1x32x32xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %100 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %101 = loom.semaphore_take %100 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %102 = loom.init_tensor %101[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %103 = arith.muli %arg4, %c8 : index
              %104 = arith.addi %arg6, %103 : index
              %105 = loom.gather ins(%94 : tensor<1x32x1xf16>) outs(%102 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %104], LR : [%c7, %104]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %92 : memref<1x32x1xf16>
              %106 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %107 = loom.semaphore_take %106 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %108 = loom.init_tensor %107[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %109 = loom.gather ins(%99 : tensor<1x32x128xf16>) outs(%108 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %104], LR : [%c7, %104]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %97 : memref<1x32x128xf16>
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
                %123 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
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
                loom.semaphore_give %101 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
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
                loom.semaphore_give %32 : memref<1x32x1xf16>
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
                loom.semaphore_give %27 : memref<1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %133 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_9, %c0_10)
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
                %30 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
                %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %27)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
                %72 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %73 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                %74 = linalg.batch_matmul ins(%26, %72 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                loom.semaphore_give %58 : memref<1x128x512xf16>
                loom.semaphore_give %23 : memref<1x32x128xf16>
                %75 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x32x512xf16>) outs(%75 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %121 = arith.maximumf %in, %out : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x1xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = arith.mulf %in_9, %cst_2 : f16
                  %122 = arith.cmpf ogt, %in, %121 : f16
                  %123 = arith.select %122, %in, %121 : f16
                  linalg.yield %123 : f16
                } -> tensor<1x32x1xf16>
                %78 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%68 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = arith.mulf %in, %cst_2 : f16
                  %122 = arith.subf %121, %in_9 : f16
                  %123 = math.exp %122 : f16
                  linalg.yield %123 : f16
                } -> tensor<1x32x512xf16>
                loom.semaphore_give %67 : memref<1x32x32xf16>
                %80 = linalg.fill ins(%cst : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x32x512xf16>) outs(%80 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %121 = arith.addf %in, %out : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = arith.subf %in, %in_9 : f16
                  %122 = math.exp %121 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %82, %81 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %121 = arith.mulf %in, %in_9 : f16
                  %122 = arith.addf %121, %in_10 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %52 : memref<1x32x1xf16>
                %84 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %55 : memref<1x32x1xf16>
                %c0_7 = arith.constant 0 : index
                %85 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %27, %c0_7)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%85], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_8, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %86 = loom.bufferize_to_tensor %70[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %87 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %88 = linalg.batch_matmul ins(%79, %86 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%87 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                loom.semaphore_give %70 : memref<1x512x128xf16>
                loom.semaphore_give %60 : memref<1x32x512xf16>
                %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %32, %84 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%32 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                  %121 = arith.mulf %in_9, %in_10 : f16
                  %122 = arith.addf %in, %121 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %65 : memref<1x32x32xf16>
                loom.semaphore_give %46 : memref<1x32x128xf16>
                %90 = linalg.copy ins(%77 : tensor<1x32x1xf16>) outs(%43 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                loom.semaphore_give %49 : memref<1x32x1xf16>
                %91 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %92 = loom.semaphore_take %91 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %93 = loom.init_tensor %92[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83, %90 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%93 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = math.log %in : f16
                  %122 = arith.addf %121, %in_9 : f16
                  linalg.yield %122 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %42 : memref<1x32x1xf16>
                %95 = loom.broadcast ins(%83 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %38 : memref<1x32x1xf16>
                %96 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %98 = loom.init_tensor %97[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %95 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%98 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %121 = arith.divf %in, %in_9 : f16
                  linalg.yield %121 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %63 : memref<1x32x32xf16>
                loom.semaphore_give %30 : memref<1x32x128xf16>
                %100 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %101 = loom.semaphore_take %100 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %102 = loom.init_tensor %101[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %103 = loom.gather ins(%94 : tensor<1x32x1xf16>) outs(%102 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%arg4, %arg6], LR : [%arg4, %arg6]) -> tensor<16x1x32x1xf16>
                loom.semaphore_give %92 : memref<1x32x1xf16>
                %104 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %105 = loom.semaphore_take %104 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %106 = loom.init_tensor %105[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %107 = loom.gather ins(%99 : tensor<1x32x128xf16>) outs(%106 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%arg4, %arg6], LR : [%arg4, %arg6]) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %97 : memref<1x32x128xf16>
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
                  %121 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
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
                  loom.semaphore_give %101 : memref<16x1x32x1xf16>
                  loom.semaphore_give %36 : memref<1x32x1xf16>
                  %124 = linalg.fill ins(%cst : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
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
                  loom.semaphore_give %34 : memref<1x32x1xf16>
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
                  loom.semaphore_give %29 : memref<1x32x128xf16>
                  %c0_9 = arith.constant 0 : index
                  %c0_10 = arith.constant 0 : index
                  %131 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%131], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %132 = loom.bufferize_to_memref %130 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                  loom.copy %132, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %109 : memref<1x32x128xf16>
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
              %31 = loom.semaphore_take %29 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %32 = loom.init_tensor %31[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %33 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %34 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %35 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %37 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %39 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %40 = loom.init_tensor %39[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %44 = loom.init_tensor %43[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %46 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %47 = loom.semaphore_take %46 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %48 = loom.init_tensor %47[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %49 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %51 = loom.init_tensor %50[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %52 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %53 = loom.semaphore_take %52 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %54 = loom.init_tensor %53[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %55 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %56 = loom.semaphore_take %55 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %57 = loom.init_tensor %56[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %58 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %59 = loom.semaphore_take %58 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %60 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %61 = loom.semaphore_take %60 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %62 = loom.init_tensor %61[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %63 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %64 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %65 = loom.init_tensor %64[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %66 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %67 = loom.init_tensor %66[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %68 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %69 = loom.init_tensor %68[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %70 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %71 = loom.semaphore_take %70 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %28)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %73 = arith.addi %arg5, %25 : index
              loom.copy %reinterpret_cast_6, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%73, %arg6], LR : [%73, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %74 = loom.bufferize_to_tensor %59[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %75 = linalg.fill ins(%cst : f16) outs(%62 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %76 = linalg.batch_matmul ins(%27, %74 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%75 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %59 : memref<1x128x512xf16>
              loom.semaphore_give %23 : memref<1x32x128xf16>
              %77 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x32x512xf16>) outs(%77 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.maximumf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%51 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in_9, %cst_2 : f16
                %124 = arith.cmpf ogt, %in, %123 : f16
                %125 = arith.select %124, %in, %123 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              %80 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%69 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %80 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%62 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in, %cst_2 : f16
                %124 = arith.subf %123, %in_9 : f16
                %125 = math.exp %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %68 : memref<1x32x32xf16>
              %82 = linalg.fill ins(%cst : f16) outs(%54 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<1x32x512xf16>) outs(%82 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.addf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %79 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%57 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.subf %in, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %84, %83 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in, %in_9 : f16
                %124 = arith.addf %123, %in_10 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %53 : memref<1x32x1xf16>
              %86 = loom.broadcast ins(%84 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %56 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %87 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %28, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%87], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%73, %arg6], LR : [%73, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %88 = loom.bufferize_to_tensor %71[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %89 = linalg.fill ins(%cst : f16) outs(%48 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %90 = linalg.batch_matmul ins(%81, %88 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%89 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %71 : memref<1x512x128xf16>
              loom.semaphore_give %61 : memref<1x32x512xf16>
              %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %33, %86 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%33 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in_9, %in_10 : f16
                %124 = arith.addf %in, %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x32x32xf16>
              loom.semaphore_give %47 : memref<1x32x128xf16>
              %92 = linalg.copy ins(%79 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %50 : memref<1x32x1xf16>
              %93 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %94 = loom.semaphore_take %93 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %95 = loom.init_tensor %94[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %92 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%95 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = math.log %in : f16
                %124 = arith.addf %123, %in_9 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %43 : memref<1x32x1xf16>
              %97 = loom.broadcast ins(%85 : tensor<1x32x1xf16>) outs(%65 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %39 : memref<1x32x1xf16>
              %98 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %99 = loom.semaphore_take %98 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %100 = loom.init_tensor %99[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %97 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%100 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.divf %in, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %64 : memref<1x32x32xf16>
              loom.semaphore_give %31 : memref<1x32x128xf16>
              %102 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %103 = loom.semaphore_take %102 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %104 = loom.init_tensor %103[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %105 = loom.gather ins(%96 : tensor<1x32x1xf16>) outs(%104 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%25, %arg6], LR : [%26, %arg6]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %94 : memref<1x32x1xf16>
              %106 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %107 = loom.semaphore_take %106 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %108 = loom.init_tensor %107[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %109 = loom.gather ins(%101 : tensor<1x32x128xf16>) outs(%108 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%25, %arg6], LR : [%26, %arg6]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %99 : memref<1x32x128xf16>
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
                %123 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%105 : tensor<16x1x32x1xf16>) outs(%123 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.maximumf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %124 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.subf %in, %in_12 : f16
                  %137 = math.exp %136 : f16
                  linalg.yield %137 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %103 : memref<16x1x32x1xf16>
                loom.semaphore_give %37 : memref<1x32x1xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x1xf16>) outs(%126 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125, %127 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.divf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %35 : memref<1x32x1xf16>
                %129 = loom.broadcast ins(%128 : tensor<16x1x32x1xf16>) outs(%121 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %114 : memref<16x1x32x1xf16>
                %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %129 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%118 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.mulf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %120 : memref<16x1x32x32xf16>
                loom.semaphore_give %107 : memref<16x1x32x128xf16>
                %131 = linalg.fill ins(%cst : f16) outs(%112 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%130 : tensor<16x1x32x128xf16>) outs(%131 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %117 : memref<16x1x32x128xf16>
                loom.semaphore_give %30 : memref<1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %133 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%133], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %134 = loom.bufferize_to_memref %132 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %135 = arith.addi %arg5, %25 : index
                loom.copy %134, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%135, %arg6], LR : [%135, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %111 : memref<1x32x128xf16>
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
              %31 = loom.semaphore_take %29 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %32 = loom.init_tensor %31[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %33 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %34 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %35 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %37 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %39 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %40 = loom.init_tensor %39[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %44 = loom.init_tensor %43[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %46 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %47 = loom.semaphore_take %46 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %48 = loom.init_tensor %47[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %49 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %51 = loom.init_tensor %50[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %52 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %53 = loom.semaphore_take %52 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %54 = loom.init_tensor %53[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %55 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %56 = loom.semaphore_take %55 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %57 = loom.init_tensor %56[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %58 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %59 = loom.semaphore_take %58 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %60 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %61 = loom.semaphore_take %60 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %62 = loom.init_tensor %61[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %63 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %64 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %65 = loom.init_tensor %64[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %66 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %67 = loom.init_tensor %66[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %68 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %69 = loom.init_tensor %68[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %70 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %71 = loom.semaphore_take %70 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %28)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %73 = arith.addi %arg5, %25 : index
              loom.copy %reinterpret_cast_6, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%73, %arg6], LR : [%73, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %74 = loom.bufferize_to_tensor %59[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %75 = linalg.fill ins(%cst : f16) outs(%62 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %76 = linalg.batch_matmul ins(%27, %74 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%75 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %59 : memref<1x128x512xf16>
              loom.semaphore_give %23 : memref<1x32x128xf16>
              %77 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x32x512xf16>) outs(%77 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.maximumf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%51 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in_9, %cst_2 : f16
                %124 = arith.cmpf ogt, %in, %123 : f16
                %125 = arith.select %124, %in, %123 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              %80 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%69 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %80 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%62 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in, %cst_2 : f16
                %124 = arith.subf %123, %in_9 : f16
                %125 = math.exp %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %68 : memref<1x32x32xf16>
              %82 = linalg.fill ins(%cst : f16) outs(%54 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<1x32x512xf16>) outs(%82 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.addf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %79 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%57 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.subf %in, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %84, %83 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in, %in_9 : f16
                %124 = arith.addf %123, %in_10 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %53 : memref<1x32x1xf16>
              %86 = loom.broadcast ins(%84 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %56 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %87 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %28, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%87], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%73, %arg6], LR : [%73, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %88 = loom.bufferize_to_tensor %71[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %89 = linalg.fill ins(%cst : f16) outs(%48 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %90 = linalg.batch_matmul ins(%81, %88 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%89 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %71 : memref<1x512x128xf16>
              loom.semaphore_give %61 : memref<1x32x512xf16>
              %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %33, %86 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%33 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in_9, %in_10 : f16
                %124 = arith.addf %in, %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x32x32xf16>
              loom.semaphore_give %47 : memref<1x32x128xf16>
              %92 = linalg.copy ins(%79 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %50 : memref<1x32x1xf16>
              %93 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %94 = loom.semaphore_take %93 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %95 = loom.init_tensor %94[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %92 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%95 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = math.log %in : f16
                %124 = arith.addf %123, %in_9 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %43 : memref<1x32x1xf16>
              %97 = loom.broadcast ins(%85 : tensor<1x32x1xf16>) outs(%65 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %39 : memref<1x32x1xf16>
              %98 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %99 = loom.semaphore_take %98 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %100 = loom.init_tensor %99[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %97 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%100 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.divf %in, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %64 : memref<1x32x32xf16>
              loom.semaphore_give %31 : memref<1x32x128xf16>
              %102 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %103 = loom.semaphore_take %102 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %104 = loom.init_tensor %103[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %105 = loom.gather ins(%96 : tensor<1x32x1xf16>) outs(%104 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%25, %arg6], LR : [%26, %arg6]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %94 : memref<1x32x1xf16>
              %106 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %107 = loom.semaphore_take %106 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %108 = loom.init_tensor %107[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %109 = loom.gather ins(%101 : tensor<1x32x128xf16>) outs(%108 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%25, %arg6], LR : [%26, %arg6]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %99 : memref<1x32x128xf16>
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
                %123 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%105 : tensor<16x1x32x1xf16>) outs(%123 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.maximumf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %124 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.subf %in, %in_12 : f16
                  %137 = math.exp %136 : f16
                  linalg.yield %137 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %103 : memref<16x1x32x1xf16>
                loom.semaphore_give %37 : memref<1x32x1xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x1xf16>) outs(%126 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125, %127 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.divf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %35 : memref<1x32x1xf16>
                %129 = loom.broadcast ins(%128 : tensor<16x1x32x1xf16>) outs(%121 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %114 : memref<16x1x32x1xf16>
                %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %129 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%118 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.mulf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %120 : memref<16x1x32x32xf16>
                loom.semaphore_give %107 : memref<16x1x32x128xf16>
                %131 = linalg.fill ins(%cst : f16) outs(%112 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%130 : tensor<16x1x32x128xf16>) outs(%131 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %117 : memref<16x1x32x128xf16>
                loom.semaphore_give %30 : memref<1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %133 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%133], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %134 = loom.bufferize_to_memref %132 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %135 = arith.addi %arg5, %25 : index
                loom.copy %134, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%135, %arg6], LR : [%135, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %111 : memref<1x32x128xf16>
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
              %28 = loom.semaphore_take %26 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %29 = loom.init_tensor %28[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %31 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %32 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %33 = loom.init_tensor %32[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %34 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %35 = loom.init_tensor %34[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %36 = loom.semaphore_take %31 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %37 = loom.init_tensor %36[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %38 = linalg.fill ins(%cst_0 : f16) outs(%37 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %39 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %40 = loom.semaphore_take %39 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %41 = loom.init_tensor %40[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %42 = linalg.fill ins(%cst_1 : f16) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %43 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %44 = loom.semaphore_take %43 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %45 = loom.init_tensor %44[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %46 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %47 = loom.semaphore_take %46 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %48 = loom.init_tensor %47[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %49 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %51 = loom.init_tensor %50[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %52 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %53 = loom.semaphore_take %52 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %54 = loom.init_tensor %53[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %55 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %56 = loom.semaphore_take %55 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %57 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %58 = loom.semaphore_take %57 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %59 = loom.init_tensor %58[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %60 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
              %61 = loom.semaphore_take %60 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %62 = loom.init_tensor %61[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %63 = loom.semaphore_take %60 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %64 = loom.init_tensor %63[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %65 = loom.semaphore_take %60 : memref<1x32x32xf16> -> memref<1x32x32xf16>
              %66 = loom.init_tensor %65[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
              %67 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %68 = loom.semaphore_take %67 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%arg7, %c0_5, %25)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%69], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %70 = arith.muli %arg4, %c8 : index
              %71 = arith.addi %arg5, %70 : index
              loom.copy %reinterpret_cast_6, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%71, %arg6], LR : [%71, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %72 = loom.bufferize_to_tensor %56[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %73 = linalg.fill ins(%cst : f16) outs(%59 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %74 = linalg.batch_matmul ins(%24, %72 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %56 : memref<1x128x512xf16>
              loom.semaphore_give %22 : memref<1x32x128xf16>
              %75 = linalg.fill ins(%cst_1 : f16) outs(%48 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x32x512xf16>) outs(%75 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.maximumf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%48 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in_9, %cst_2 : f16
                %124 = arith.cmpf ogt, %in, %123 : f16
                %125 = arith.select %124, %in, %123 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x1xf16>
              %78 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%59 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.mulf %in, %cst_2 : f16
                %124 = arith.subf %123, %in_9 : f16
                %125 = math.exp %124 : f16
                linalg.yield %125 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %65 : memref<1x32x32xf16>
              %80 = linalg.fill ins(%cst : f16) outs(%51 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x32x512xf16>) outs(%80 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %123 = arith.addf %in, %out : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%54 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.subf %in, %in_9 : f16
                %124 = math.exp %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %82, %81 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in, %in_9 : f16
                %124 = arith.addf %123, %in_10 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %50 : memref<1x32x1xf16>
              %84 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %53 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %85 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%arg7, %25, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%85], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%71, %arg6], LR : [%71, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %86 = loom.bufferize_to_tensor %68[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %87 = linalg.fill ins(%cst : f16) outs(%45 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %88 = linalg.batch_matmul ins(%79, %86 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%87 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %68 : memref<1x512x128xf16>
              loom.semaphore_give %58 : memref<1x32x512xf16>
              %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %30, %84 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %123 = arith.mulf %in_9, %in_10 : f16
                %124 = arith.addf %in, %123 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %63 : memref<1x32x32xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              %90 = linalg.copy ins(%77 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %47 : memref<1x32x1xf16>
              %91 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %92 = loom.semaphore_take %91 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %93 = loom.init_tensor %92[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83, %90 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%93 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = math.log %in : f16
                %124 = arith.addf %123, %in_9 : f16
                linalg.yield %124 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %95 = loom.broadcast ins(%83 : tensor<1x32x1xf16>) outs(%62 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %96 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %97 = loom.semaphore_take %96 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %98 = loom.init_tensor %97[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %95 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%98 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %123 = arith.divf %in, %in_9 : f16
                linalg.yield %123 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %61 : memref<1x32x32xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %100 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %101 = loom.semaphore_take %100 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %102 = loom.init_tensor %101[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %103 = arith.muli %arg4, %c8 : index
              %104 = arith.addi %103, %c7 : index
              %105 = loom.gather ins(%94 : tensor<1x32x1xf16>) outs(%102 : tensor<16x1x32x1xf16>) across(%arg5 : index) region : (UL : [%103, %arg6], LR : [%104, %arg6]) -> tensor<16x1x32x1xf16>
              loom.semaphore_give %92 : memref<1x32x1xf16>
              %106 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %107 = loom.semaphore_take %106 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %108 = loom.init_tensor %107[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %109 = loom.gather ins(%99 : tensor<1x32x128xf16>) outs(%108 : tensor<16x1x32x128xf16>) across(%arg5 : index) region : (UL : [%103, %arg6], LR : [%104, %arg6]) -> tensor<16x1x32x128xf16>
              loom.semaphore_give %97 : memref<1x32x128xf16>
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
                %123 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%105 : tensor<16x1x32x1xf16>) outs(%123 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.maximumf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %124 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.subf %in, %in_12 : f16
                  %137 = math.exp %136 : f16
                  linalg.yield %137 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %101 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x1xf16>) outs(%126 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x1xf16>
                %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%125, %127 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%115 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.divf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %32 : memref<1x32x1xf16>
                %129 = loom.broadcast ins(%128 : tensor<16x1x32x1xf16>) outs(%121 : tensor<16x1x32x32xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %114 : memref<16x1x32x1xf16>
                %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %129 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%118 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %136 = arith.mulf %in, %in_12 : f16
                  linalg.yield %136 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %120 : memref<16x1x32x32xf16>
                loom.semaphore_give %107 : memref<16x1x32x128xf16>
                %131 = linalg.fill ins(%cst : f16) outs(%112 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%130 : tensor<16x1x32x128xf16>) outs(%131 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %136 = arith.addf %in, %out : f16
                  linalg.yield %136 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %117 : memref<16x1x32x128xf16>
                loom.semaphore_give %27 : memref<1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %133 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%133], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %134 = loom.bufferize_to_memref %132 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %135 = arith.addi %arg5, %103 : index
                loom.copy %134, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%135, %arg6], LR : [%135, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %111 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
