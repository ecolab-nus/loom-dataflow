module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y1y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_x_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (1) {
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %21 = arith.muli %arg5, %c1024 : index
              %22 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %23 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %24 = loom.init_tensor %23[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %25 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %26 = loom.init_tensor %25[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %27 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %28 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %29 = loom.init_tensor %28[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %c0_3 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %21, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %31 = arith.addi %arg6, %arg4 : index
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
              %32 = loom.bufferize_to_tensor %27[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %33 = loom.sync ins(%32 : tensor<1x1024x128xf16>) outs(%29 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %27 : memref<1x1024x128xf16>
              %34 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %35 = loom.semaphore_take %34 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %36 = loom.init_tensor %35[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %40 = loom.init_tensor %39[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %44 = loom.init_tensor %43[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %46 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %47 = loom.semaphore_take %46 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %48 = loom.init_tensor %47[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %49 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %51 = loom.init_tensor %50[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %52 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %53 = loom.semaphore_take %52 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %54 = loom.init_tensor %53[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %55 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %56 = loom.semaphore_take %55 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %57 = loom.init_tensor %56[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %58 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
              %59 = loom.semaphore_take %58 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %60 = loom.init_tensor %59[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %61 = loom.semaphore_take %58 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %62 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
              %63 = loom.semaphore_take %62 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
              %64 = loom.init_tensor %63[1, 1024, 64] : memref<1x1024x64xf16> -> tensor<1x1024x64xf16>
              %65 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
              %66 = loom.semaphore_take %65 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %67 = loom.init_tensor %66[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %68 = loom.semaphore_take %65 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %69 = loom.init_tensor %68[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %70 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
              %71 = loom.semaphore_take %70 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %72 = loom.init_tensor %71[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %73 = loom.semaphore_take %70 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %74:3 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %45, %arg10 = %41, %arg11 = %37) -> (tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>) {
                %80 = arith.muli %arg8, %c64 : index
                %c0_6 = arith.constant 0 : index
                %81 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%20, %c0_6, %80)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%81], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %31], LR : [%c7, %31]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
                %82 = loom.bufferize_to_tensor %61[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
                %83 = loom.sync ins(%82 : tensor<1x128x64xf16>) outs(%60 : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>
                loom.semaphore_give %61 : memref<1x128x64xf16>
                %84 = linalg.fill ins(%cst : f16) outs(%64 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                %85 = linalg.batch_matmul ins(%33, %83 : tensor<1x1024x128xf16>, tensor<1x128x64xf16>) outs(%84 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                loom.semaphore_give %59 : memref<1x128x64xf16>
                %86 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<1x1024x64xf16>) outs(%86 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %102 = arith.maximumf %in, %out : f16
                  linalg.yield %102 : f16
                } -> tensor<1x1024x1xf16>
                %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %87 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%51 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %102 = arith.mulf %in_10, %cst_2 : f16
                  %103 = arith.cmpf ogt, %in, %102 : f16
                  %104 = arith.select %103, %in, %102 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x1xf16>
                %89 = loom.broadcast ins(%88 : tensor<1x1024x1xf16>) outs(%69 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x64xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %89 : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>) outs(%64 : tensor<1x1024x64xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %102 = arith.mulf %in, %cst_2 : f16
                  %103 = arith.subf %102, %in_10 : f16
                  %104 = math.exp %103 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x64xf16>
                loom.semaphore_give %68 : memref<1x1024x32xf16>
                %91 = linalg.fill ins(%cst : f16) outs(%54 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%90 : tensor<1x1024x64xf16>) outs(%91 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %102 = arith.addf %in, %out : f16
                  linalg.yield %102 : f16
                } -> tensor<1x1024x1xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %88 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%57 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %102 = arith.subf %in, %in_10 : f16
                  %103 = math.exp %102 : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x1xf16>
                %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %93, %92 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%arg10 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %102 = arith.mulf %in, %in_10 : f16
                  %103 = arith.addf %102, %in_11 : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x1xf16>
                loom.semaphore_give %53 : memref<1x1024x1xf16>
                %c0_8 = arith.constant 0 : index
                %95 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %80, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%95], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %31], LR : [%c7, %31]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
                %96 = loom.bufferize_to_tensor %73[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
                %97 = loom.sync ins(%96 : tensor<1x64x128xf16>) outs(%72 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
                loom.semaphore_give %73 : memref<1x64x128xf16>
                %98 = linalg.fill ins(%cst : f16) outs(%48 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                %99 = linalg.batch_matmul ins(%90, %97 : tensor<1x1024x64xf16>, tensor<1x64x128xf16>) outs(%98 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                loom.semaphore_give %71 : memref<1x64x128xf16>
                loom.semaphore_give %63 : memref<1x1024x64xf16>
                %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%99, %arg11, %93 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x1xf16>) outs(%arg11 : tensor<1x1024x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %102 = arith.mulf %in_10, %in_11 : f16
                  %103 = arith.addf %in, %102 : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x128xf16>
                loom.semaphore_give %56 : memref<1x1024x1xf16>
                loom.semaphore_give %47 : memref<1x1024x128xf16>
                %101 = linalg.copy ins(%88 : tensor<1x1024x1xf16>) outs(%arg9 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                loom.semaphore_give %50 : memref<1x1024x1xf16>
                scf.yield %101, %94, %100 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>
              }
              loom.semaphore_give %43 : memref<1x1024x1xf16>
              loom.semaphore_give %28 : memref<1x1024x128xf16>
              %75 = loom.broadcast ins(%74#1 : tensor<1x1024x1xf16>) outs(%67 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x128xf16>
              loom.semaphore_give %39 : memref<1x1024x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74#2, %75 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>) outs(%26 : tensor<1x1024x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %80 = arith.divf %in, %in_6 : f16
                linalg.yield %80 : f16
              } -> tensor<1x1024x128xf16>
              loom.semaphore_give %66 : memref<1x1024x32xf16>
              loom.semaphore_give %35 : memref<1x1024x128xf16>
              %77 = loom.sync ins(%76 : tensor<1x1024x128xf16>) outs(%24 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %25 : memref<1x1024x128xf16>
              %c0_4 = arith.constant 0 : index
              %78 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %21, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%78], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %79 = loom.bufferize_to_memref %77 : tensor<1x1024x128xf16> -> memref<1x1024x128xf16>
              loom.copy %79, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %23 : memref<1x1024x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y2y4__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc2_dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (2) {
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %22 = arith.muli %21, %c1024 : index
              %23 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %25 = loom.init_tensor %24[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %26 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %27 = loom.init_tensor %26[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %28 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %29 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %30 = loom.init_tensor %29[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %c0_3 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %22, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %32 = arith.muli %arg4, %c2 : index
              %33 = arith.addi %arg6, %32 : index
              loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
              %34 = loom.bufferize_to_tensor %28[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %35 = loom.sync ins(%34 : tensor<1x1024x128xf16>) outs(%30 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %28 : memref<1x1024x128xf16>
              %36 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %37 = loom.semaphore_take %36 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %38 = loom.init_tensor %37[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              %40 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %41 = loom.semaphore_take %40 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %42 = loom.init_tensor %41[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %44 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %45 = loom.semaphore_take %44 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %46 = loom.init_tensor %45[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %48 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %49 = loom.semaphore_take %48 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %50 = loom.init_tensor %49[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %51 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %53 = loom.init_tensor %52[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %54 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %56 = loom.init_tensor %55[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %57 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %58 = loom.semaphore_take %57 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %59 = loom.init_tensor %58[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %60 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
              %61 = loom.semaphore_take %60 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %62 = loom.init_tensor %61[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %63 = loom.semaphore_take %60 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %64 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
              %65 = loom.semaphore_take %64 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
              %66 = loom.init_tensor %65[1, 1024, 64] : memref<1x1024x64xf16> -> tensor<1x1024x64xf16>
              %67 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
              %68 = loom.semaphore_take %67 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %69 = loom.init_tensor %68[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %70 = loom.semaphore_take %67 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %71 = loom.init_tensor %70[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %72 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
              %73 = loom.semaphore_take %72 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %74 = loom.init_tensor %73[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %75 = loom.semaphore_take %72 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %76:3 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %47, %arg10 = %43, %arg11 = %39) -> (tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>) {
                %82 = arith.muli %arg8, %c64 : index
                %c0_6 = arith.constant 0 : index
                %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%20, %c0_6, %82)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%83], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
                %84 = arith.addi %32, %c1 : index
                loom.copy %reinterpret_cast_7, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 2] region : (UL : [%c0, %32], LR : [%c7, %84]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
                %85 = loom.bufferize_to_tensor %63[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
                %86 = loom.sync ins(%85 : tensor<1x128x64xf16>) outs(%62 : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>
                loom.semaphore_give %63 : memref<1x128x64xf16>
                %87 = linalg.fill ins(%cst : f16) outs(%66 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                %88 = linalg.batch_matmul ins(%35, %86 : tensor<1x1024x128xf16>, tensor<1x128x64xf16>) outs(%87 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                loom.semaphore_give %61 : memref<1x128x64xf16>
                %89 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%88 : tensor<1x1024x64xf16>) outs(%89 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %105 = arith.maximumf %in, %out : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %90 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%53 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.mulf %in_10, %cst_2 : f16
                  %106 = arith.cmpf ogt, %in, %105 : f16
                  %107 = arith.select %106, %in, %105 : f16
                  linalg.yield %107 : f16
                } -> tensor<1x1024x1xf16>
                %92 = loom.broadcast ins(%91 : tensor<1x1024x1xf16>) outs(%71 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x64xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %92 : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>) outs(%66 : tensor<1x1024x64xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.mulf %in, %cst_2 : f16
                  %106 = arith.subf %105, %in_10 : f16
                  %107 = math.exp %106 : f16
                  linalg.yield %107 : f16
                } -> tensor<1x1024x64xf16>
                loom.semaphore_give %70 : memref<1x1024x32xf16>
                %94 = linalg.fill ins(%cst : f16) outs(%56 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%93 : tensor<1x1024x64xf16>) outs(%94 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %105 = arith.addf %in, %out : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %91 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%59 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.subf %in, %in_10 : f16
                  %106 = math.exp %105 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x1xf16>
                %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96, %95 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%arg10 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %105 = arith.mulf %in, %in_10 : f16
                  %106 = arith.addf %105, %in_11 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x1xf16>
                loom.semaphore_give %55 : memref<1x1024x1xf16>
                %c0_8 = arith.constant 0 : index
                %98 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %82, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 2] region : (UL : [%c0, %32], LR : [%c7, %84]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
                %99 = loom.bufferize_to_tensor %75[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
                %100 = loom.sync ins(%99 : tensor<1x64x128xf16>) outs(%74 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
                loom.semaphore_give %75 : memref<1x64x128xf16>
                %101 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                %102 = linalg.batch_matmul ins(%93, %100 : tensor<1x1024x64xf16>, tensor<1x64x128xf16>) outs(%101 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                loom.semaphore_give %73 : memref<1x64x128xf16>
                loom.semaphore_give %65 : memref<1x1024x64xf16>
                %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102, %arg11, %96 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x1xf16>) outs(%arg11 : tensor<1x1024x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %105 = arith.mulf %in_10, %in_11 : f16
                  %106 = arith.addf %in, %105 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x128xf16>
                loom.semaphore_give %58 : memref<1x1024x1xf16>
                loom.semaphore_give %49 : memref<1x1024x128xf16>
                %104 = linalg.copy ins(%91 : tensor<1x1024x1xf16>) outs(%arg9 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                loom.semaphore_give %52 : memref<1x1024x1xf16>
                scf.yield %104, %97, %103 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>
              }
              loom.semaphore_give %45 : memref<1x1024x1xf16>
              loom.semaphore_give %29 : memref<1x1024x128xf16>
              %77 = loom.broadcast ins(%76#1 : tensor<1x1024x1xf16>) outs(%69 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x128xf16>
              loom.semaphore_give %41 : memref<1x1024x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76#2, %77 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>) outs(%27 : tensor<1x1024x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %82 = arith.divf %in, %in_6 : f16
                linalg.yield %82 : f16
              } -> tensor<1x1024x128xf16>
              loom.semaphore_give %68 : memref<1x1024x32xf16>
              loom.semaphore_give %37 : memref<1x1024x128xf16>
              %79 = loom.sync ins(%78 : tensor<1x1024x128xf16>) outs(%25 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %26 : memref<1x1024x128xf16>
              %c0_4 = arith.constant 0 : index
              %80 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %22, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%80], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %81 = loom.bufferize_to_memref %79 : tensor<1x1024x128xf16> -> memref<1x1024x128xf16>
              loom.copy %81, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %24 : memref<1x1024x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y4y2__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc4_dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c16 = arith.constant 16 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (4) {
            scf.for %arg7 = %c0 to %c16 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %22 = arith.muli %21, %c1024 : index
              %23 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %25 = loom.init_tensor %24[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %26 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %27 = loom.init_tensor %26[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %28 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %29 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %30 = loom.init_tensor %29[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %c0_3 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %22, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %32 = arith.muli %arg4, %c4 : index
              %33 = arith.addi %arg6, %32 : index
              loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
              %34 = loom.bufferize_to_tensor %28[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %35 = loom.sync ins(%34 : tensor<1x1024x128xf16>) outs(%30 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %28 : memref<1x1024x128xf16>
              %36 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %37 = loom.semaphore_take %36 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %38 = loom.init_tensor %37[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              %40 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %41 = loom.semaphore_take %40 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %42 = loom.init_tensor %41[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %44 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %45 = loom.semaphore_take %44 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %46 = loom.init_tensor %45[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %48 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %49 = loom.semaphore_take %48 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %50 = loom.init_tensor %49[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %51 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %53 = loom.init_tensor %52[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %54 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %56 = loom.init_tensor %55[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %57 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %58 = loom.semaphore_take %57 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %59 = loom.init_tensor %58[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %60 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
              %61 = loom.semaphore_take %60 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %62 = loom.init_tensor %61[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %63 = loom.semaphore_take %60 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %64 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
              %65 = loom.semaphore_take %64 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
              %66 = loom.init_tensor %65[1, 1024, 64] : memref<1x1024x64xf16> -> tensor<1x1024x64xf16>
              %67 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
              %68 = loom.semaphore_take %67 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %69 = loom.init_tensor %68[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %70 = loom.semaphore_take %67 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %71 = loom.init_tensor %70[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %72 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
              %73 = loom.semaphore_take %72 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %74 = loom.init_tensor %73[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %75 = loom.semaphore_take %72 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %76:3 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %47, %arg10 = %43, %arg11 = %39) -> (tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>) {
                %82 = arith.muli %arg8, %c64 : index
                %c0_6 = arith.constant 0 : index
                %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%20, %c0_6, %82)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%83], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
                %84 = arith.addi %32, %c3 : index
                loom.copy %reinterpret_cast_7, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %84]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
                %85 = loom.bufferize_to_tensor %63[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
                %86 = loom.sync ins(%85 : tensor<1x128x64xf16>) outs(%62 : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>
                loom.semaphore_give %63 : memref<1x128x64xf16>
                %87 = linalg.fill ins(%cst : f16) outs(%66 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                %88 = linalg.batch_matmul ins(%35, %86 : tensor<1x1024x128xf16>, tensor<1x128x64xf16>) outs(%87 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                loom.semaphore_give %61 : memref<1x128x64xf16>
                %89 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%88 : tensor<1x1024x64xf16>) outs(%89 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %105 = arith.maximumf %in, %out : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %90 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%53 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.mulf %in_10, %cst_2 : f16
                  %106 = arith.cmpf ogt, %in, %105 : f16
                  %107 = arith.select %106, %in, %105 : f16
                  linalg.yield %107 : f16
                } -> tensor<1x1024x1xf16>
                %92 = loom.broadcast ins(%91 : tensor<1x1024x1xf16>) outs(%71 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x64xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %92 : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>) outs(%66 : tensor<1x1024x64xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.mulf %in, %cst_2 : f16
                  %106 = arith.subf %105, %in_10 : f16
                  %107 = math.exp %106 : f16
                  linalg.yield %107 : f16
                } -> tensor<1x1024x64xf16>
                loom.semaphore_give %70 : memref<1x1024x32xf16>
                %94 = linalg.fill ins(%cst : f16) outs(%56 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%93 : tensor<1x1024x64xf16>) outs(%94 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %105 = arith.addf %in, %out : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %91 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%59 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.subf %in, %in_10 : f16
                  %106 = math.exp %105 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x1xf16>
                %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96, %95 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%arg10 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %105 = arith.mulf %in, %in_10 : f16
                  %106 = arith.addf %105, %in_11 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x1xf16>
                loom.semaphore_give %55 : memref<1x1024x1xf16>
                %c0_8 = arith.constant 0 : index
                %98 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %82, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %84]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
                %99 = loom.bufferize_to_tensor %75[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
                %100 = loom.sync ins(%99 : tensor<1x64x128xf16>) outs(%74 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
                loom.semaphore_give %75 : memref<1x64x128xf16>
                %101 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                %102 = linalg.batch_matmul ins(%93, %100 : tensor<1x1024x64xf16>, tensor<1x64x128xf16>) outs(%101 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                loom.semaphore_give %73 : memref<1x64x128xf16>
                loom.semaphore_give %65 : memref<1x1024x64xf16>
                %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102, %arg11, %96 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x1xf16>) outs(%arg11 : tensor<1x1024x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %105 = arith.mulf %in_10, %in_11 : f16
                  %106 = arith.addf %in, %105 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x128xf16>
                loom.semaphore_give %58 : memref<1x1024x1xf16>
                loom.semaphore_give %49 : memref<1x1024x128xf16>
                %104 = linalg.copy ins(%91 : tensor<1x1024x1xf16>) outs(%arg9 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                loom.semaphore_give %52 : memref<1x1024x1xf16>
                scf.yield %104, %97, %103 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>
              }
              loom.semaphore_give %45 : memref<1x1024x1xf16>
              loom.semaphore_give %29 : memref<1x1024x128xf16>
              %77 = loom.broadcast ins(%76#1 : tensor<1x1024x1xf16>) outs(%69 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x128xf16>
              loom.semaphore_give %41 : memref<1x1024x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76#2, %77 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>) outs(%27 : tensor<1x1024x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %82 = arith.divf %in, %in_6 : f16
                linalg.yield %82 : f16
              } -> tensor<1x1024x128xf16>
              loom.semaphore_give %68 : memref<1x1024x32xf16>
              loom.semaphore_give %37 : memref<1x1024x128xf16>
              %79 = loom.sync ins(%78 : tensor<1x1024x128xf16>) outs(%25 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %26 : memref<1x1024x128xf16>
              %c0_4 = arith.constant 0 : index
              %80 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %22, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%80], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %81 = loom.bufferize_to_memref %79 : tensor<1x1024x128xf16> -> memref<1x1024x128xf16>
              loom.copy %81, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %24 : memref<1x1024x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y8y1__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level1_bc8_dim_x_level0_bc8_dim_y_level1_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c32 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %21 = arith.muli %20, %c1024 : index
              %22 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %23 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %24 = loom.init_tensor %23[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %25 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %26 = loom.init_tensor %25[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %27 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %28 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %29 = loom.init_tensor %28[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %c0_3 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %21, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %31 = arith.muli %arg4, %c8 : index
              %32 = arith.addi %arg6, %31 : index
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %32], LR : [%arg5, %32]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
              %33 = loom.bufferize_to_tensor %27[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %34 = loom.sync ins(%33 : tensor<1x1024x128xf16>) outs(%29 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %27 : memref<1x1024x128xf16>
              %35 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %36 = loom.semaphore_take %35 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %37 = loom.init_tensor %36[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              %39 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %40 = loom.semaphore_take %39 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %41 = loom.init_tensor %40[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %42 = linalg.fill ins(%cst_0 : f16) outs(%41 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %43 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %44 = loom.semaphore_take %43 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %45 = loom.init_tensor %44[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %46 = linalg.fill ins(%cst_1 : f16) outs(%45 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %47 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %48 = loom.semaphore_take %47 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %49 = loom.init_tensor %48[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %50 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %51 = loom.semaphore_take %50 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %52 = loom.init_tensor %51[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %53 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %54 = loom.semaphore_take %53 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %55 = loom.init_tensor %54[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %56 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %57 = loom.semaphore_take %56 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %58 = loom.init_tensor %57[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %59 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
              %60 = loom.semaphore_take %59 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %61 = loom.init_tensor %60[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %62 = loom.semaphore_take %59 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %63 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
              %64 = loom.semaphore_take %63 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
              %65 = loom.init_tensor %64[1, 1024, 64] : memref<1x1024x64xf16> -> tensor<1x1024x64xf16>
              %66 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
              %67 = loom.semaphore_take %66 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %68 = loom.init_tensor %67[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %69 = loom.semaphore_take %66 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %70 = loom.init_tensor %69[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %71 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
              %72 = loom.semaphore_take %71 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %73 = loom.init_tensor %72[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %74 = loom.semaphore_take %71 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %75:3 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %46, %arg10 = %42, %arg11 = %38) -> (tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>) {
                %81 = arith.muli %arg8, %c64 : index
                %c0_6 = arith.constant 0 : index
                %82 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%arg7, %c0_6, %81)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%82], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
                %83 = loom.bufferize_to_tensor %62[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
                %84 = loom.sync ins(%83 : tensor<1x128x64xf16>) outs(%61 : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>
                loom.semaphore_give %62 : memref<1x128x64xf16>
                %85 = linalg.fill ins(%cst : f16) outs(%65 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                %86 = linalg.batch_matmul ins(%34, %84 : tensor<1x1024x128xf16>, tensor<1x128x64xf16>) outs(%85 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                loom.semaphore_give %60 : memref<1x128x64xf16>
                %87 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%86 : tensor<1x1024x64xf16>) outs(%87 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %103 = arith.maximumf %in, %out : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x1xf16>
                %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %88 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%52 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %103 = arith.mulf %in_10, %cst_2 : f16
                  %104 = arith.cmpf ogt, %in, %103 : f16
                  %105 = arith.select %104, %in, %103 : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %90 = loom.broadcast ins(%89 : tensor<1x1024x1xf16>) outs(%70 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x64xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %90 : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>) outs(%65 : tensor<1x1024x64xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %103 = arith.mulf %in, %cst_2 : f16
                  %104 = arith.subf %103, %in_10 : f16
                  %105 = math.exp %104 : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x64xf16>
                loom.semaphore_give %69 : memref<1x1024x32xf16>
                %92 = linalg.fill ins(%cst : f16) outs(%55 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%91 : tensor<1x1024x64xf16>) outs(%92 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %103 = arith.addf %in, %out : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x1xf16>
                %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %89 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%58 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %103 = arith.subf %in, %in_10 : f16
                  %104 = math.exp %103 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x1xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %94, %93 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%arg10 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %103 = arith.mulf %in, %in_10 : f16
                  %104 = arith.addf %103, %in_11 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x1xf16>
                loom.semaphore_give %54 : memref<1x1024x1xf16>
                %c0_8 = arith.constant 0 : index
                %96 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %81, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%96], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
                %97 = loom.bufferize_to_tensor %74[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
                %98 = loom.sync ins(%97 : tensor<1x64x128xf16>) outs(%73 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
                loom.semaphore_give %74 : memref<1x64x128xf16>
                %99 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                %100 = linalg.batch_matmul ins(%91, %98 : tensor<1x1024x64xf16>, tensor<1x64x128xf16>) outs(%99 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                loom.semaphore_give %72 : memref<1x64x128xf16>
                loom.semaphore_give %64 : memref<1x1024x64xf16>
                %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100, %arg11, %94 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x1xf16>) outs(%arg11 : tensor<1x1024x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %103 = arith.mulf %in_10, %in_11 : f16
                  %104 = arith.addf %in, %103 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x128xf16>
                loom.semaphore_give %57 : memref<1x1024x1xf16>
                loom.semaphore_give %48 : memref<1x1024x128xf16>
                %102 = linalg.copy ins(%89 : tensor<1x1024x1xf16>) outs(%arg9 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                loom.semaphore_give %51 : memref<1x1024x1xf16>
                scf.yield %102, %95, %101 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>
              }
              loom.semaphore_give %44 : memref<1x1024x1xf16>
              loom.semaphore_give %28 : memref<1x1024x128xf16>
              %76 = loom.broadcast ins(%75#1 : tensor<1x1024x1xf16>) outs(%68 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x128xf16>
              loom.semaphore_give %40 : memref<1x1024x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75#2, %76 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>) outs(%26 : tensor<1x1024x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %81 = arith.divf %in, %in_6 : f16
                linalg.yield %81 : f16
              } -> tensor<1x1024x128xf16>
              loom.semaphore_give %67 : memref<1x1024x32xf16>
              loom.semaphore_give %36 : memref<1x1024x128xf16>
              %78 = loom.sync ins(%77 : tensor<1x1024x128xf16>) outs(%24 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %25 : memref<1x1024x128xf16>
              %c0_4 = arith.constant 0 : index
              %79 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %21, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%79], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %80 = loom.bufferize_to_memref %78 : tensor<1x1024x128xf16> -> memref<1x1024x128xf16>
              loom.copy %80, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %32], LR : [%arg5, %32]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %23 : memref<1x1024x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x1x8_y8__d0i1_d1i1_d2i0__f01__n_dim_y_level0_bc8_dim_y_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (1) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %21 = arith.muli %arg6, %c1024 : index
              %22 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %23 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %24 = loom.init_tensor %23[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %25 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %26 = loom.init_tensor %25[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %27 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %28 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %29 = loom.init_tensor %28[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %c0_3 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %21, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %31 = arith.addi %arg5, %arg4 : index
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%31, %arg6], LR : [%31, %arg6]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
              %32 = loom.bufferize_to_tensor %27[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %33 = loom.sync ins(%32 : tensor<1x1024x128xf16>) outs(%29 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %27 : memref<1x1024x128xf16>
              %34 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %35 = loom.semaphore_take %34 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %36 = loom.init_tensor %35[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %40 = loom.init_tensor %39[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %44 = loom.init_tensor %43[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %46 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %47 = loom.semaphore_take %46 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %48 = loom.init_tensor %47[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %49 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %51 = loom.init_tensor %50[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %52 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %53 = loom.semaphore_take %52 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %54 = loom.init_tensor %53[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %55 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %56 = loom.semaphore_take %55 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %57 = loom.init_tensor %56[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %58 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
              %59 = loom.semaphore_take %58 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %60 = loom.init_tensor %59[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %61 = loom.semaphore_take %58 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %62 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
              %63 = loom.semaphore_take %62 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
              %64 = loom.init_tensor %63[1, 1024, 64] : memref<1x1024x64xf16> -> tensor<1x1024x64xf16>
              %65 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
              %66 = loom.semaphore_take %65 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %67 = loom.init_tensor %66[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %68 = loom.semaphore_take %65 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %69 = loom.init_tensor %68[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %70 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
              %71 = loom.semaphore_take %70 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %72 = loom.init_tensor %71[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %73 = loom.semaphore_take %70 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %74:3 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %45, %arg10 = %41, %arg11 = %37) -> (tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>) {
                %80 = arith.muli %arg8, %c64 : index
                %c0_6 = arith.constant 0 : index
                %81 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%20, %c0_6, %80)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%81], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%31, %c0], LR : [%31, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
                %82 = loom.bufferize_to_tensor %61[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
                %83 = loom.sync ins(%82 : tensor<1x128x64xf16>) outs(%60 : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>
                loom.semaphore_give %61 : memref<1x128x64xf16>
                %84 = linalg.fill ins(%cst : f16) outs(%64 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                %85 = linalg.batch_matmul ins(%33, %83 : tensor<1x1024x128xf16>, tensor<1x128x64xf16>) outs(%84 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                loom.semaphore_give %59 : memref<1x128x64xf16>
                %86 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<1x1024x64xf16>) outs(%86 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %102 = arith.maximumf %in, %out : f16
                  linalg.yield %102 : f16
                } -> tensor<1x1024x1xf16>
                %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %87 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%51 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %102 = arith.mulf %in_10, %cst_2 : f16
                  %103 = arith.cmpf ogt, %in, %102 : f16
                  %104 = arith.select %103, %in, %102 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x1xf16>
                %89 = loom.broadcast ins(%88 : tensor<1x1024x1xf16>) outs(%69 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x64xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %89 : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>) outs(%64 : tensor<1x1024x64xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %102 = arith.mulf %in, %cst_2 : f16
                  %103 = arith.subf %102, %in_10 : f16
                  %104 = math.exp %103 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x64xf16>
                loom.semaphore_give %68 : memref<1x1024x32xf16>
                %91 = linalg.fill ins(%cst : f16) outs(%54 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%90 : tensor<1x1024x64xf16>) outs(%91 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %102 = arith.addf %in, %out : f16
                  linalg.yield %102 : f16
                } -> tensor<1x1024x1xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %88 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%57 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %102 = arith.subf %in, %in_10 : f16
                  %103 = math.exp %102 : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x1xf16>
                %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %93, %92 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%arg10 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %102 = arith.mulf %in, %in_10 : f16
                  %103 = arith.addf %102, %in_11 : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x1xf16>
                loom.semaphore_give %53 : memref<1x1024x1xf16>
                %c0_8 = arith.constant 0 : index
                %95 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %80, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%95], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%31, %c0], LR : [%31, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
                %96 = loom.bufferize_to_tensor %73[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
                %97 = loom.sync ins(%96 : tensor<1x64x128xf16>) outs(%72 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
                loom.semaphore_give %73 : memref<1x64x128xf16>
                %98 = linalg.fill ins(%cst : f16) outs(%48 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                %99 = linalg.batch_matmul ins(%90, %97 : tensor<1x1024x64xf16>, tensor<1x64x128xf16>) outs(%98 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                loom.semaphore_give %71 : memref<1x64x128xf16>
                loom.semaphore_give %63 : memref<1x1024x64xf16>
                %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%99, %arg11, %93 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x1xf16>) outs(%arg11 : tensor<1x1024x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %102 = arith.mulf %in_10, %in_11 : f16
                  %103 = arith.addf %in, %102 : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x128xf16>
                loom.semaphore_give %56 : memref<1x1024x1xf16>
                loom.semaphore_give %47 : memref<1x1024x128xf16>
                %101 = linalg.copy ins(%88 : tensor<1x1024x1xf16>) outs(%arg9 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                loom.semaphore_give %50 : memref<1x1024x1xf16>
                scf.yield %101, %94, %100 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>
              }
              loom.semaphore_give %43 : memref<1x1024x1xf16>
              loom.semaphore_give %28 : memref<1x1024x128xf16>
              %75 = loom.broadcast ins(%74#1 : tensor<1x1024x1xf16>) outs(%67 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x128xf16>
              loom.semaphore_give %39 : memref<1x1024x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74#2, %75 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>) outs(%26 : tensor<1x1024x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %80 = arith.divf %in, %in_6 : f16
                linalg.yield %80 : f16
              } -> tensor<1x1024x128xf16>
              loom.semaphore_give %66 : memref<1x1024x32xf16>
              loom.semaphore_give %35 : memref<1x1024x128xf16>
              %77 = loom.sync ins(%76 : tensor<1x1024x128xf16>) outs(%24 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %25 : memref<1x1024x128xf16>
              %c0_4 = arith.constant 0 : index
              %78 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %21, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%78], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %79 = loom.bufferize_to_memref %77 : tensor<1x1024x128xf16> -> memref<1x1024x128xf16>
              loom.copy %79, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%31, %arg6], LR : [%31, %arg6]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %23 : memref<1x1024x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x2x4_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc2_dim_y_level0_bc8_dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (2) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg5, %arg6)
              %22 = arith.muli %21, %c1024 : index
              %23 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %25 = loom.init_tensor %24[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %26 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %27 = loom.init_tensor %26[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %28 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %29 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %30 = loom.init_tensor %29[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %c0_3 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %22, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %32 = arith.muli %arg4, %c2 : index
              %33 = arith.addi %arg5, %32 : index
              loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
              %34 = loom.bufferize_to_tensor %28[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %35 = loom.sync ins(%34 : tensor<1x1024x128xf16>) outs(%30 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %28 : memref<1x1024x128xf16>
              %36 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %37 = loom.semaphore_take %36 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %38 = loom.init_tensor %37[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              %40 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %41 = loom.semaphore_take %40 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %42 = loom.init_tensor %41[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %44 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %45 = loom.semaphore_take %44 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %46 = loom.init_tensor %45[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %48 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %49 = loom.semaphore_take %48 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %50 = loom.init_tensor %49[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %51 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %53 = loom.init_tensor %52[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %54 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %56 = loom.init_tensor %55[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %57 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %58 = loom.semaphore_take %57 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %59 = loom.init_tensor %58[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %60 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
              %61 = loom.semaphore_take %60 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %62 = loom.init_tensor %61[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %63 = loom.semaphore_take %60 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %64 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
              %65 = loom.semaphore_take %64 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
              %66 = loom.init_tensor %65[1, 1024, 64] : memref<1x1024x64xf16> -> tensor<1x1024x64xf16>
              %67 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
              %68 = loom.semaphore_take %67 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %69 = loom.init_tensor %68[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %70 = loom.semaphore_take %67 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %71 = loom.init_tensor %70[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %72 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
              %73 = loom.semaphore_take %72 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %74 = loom.init_tensor %73[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %75 = loom.semaphore_take %72 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %76:3 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %47, %arg10 = %43, %arg11 = %39) -> (tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>) {
                %82 = arith.muli %arg8, %c64 : index
                %c0_6 = arith.constant 0 : index
                %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%20, %c0_6, %82)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%83], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
                %84 = arith.addi %32, %c1 : index
                loom.copy %reinterpret_cast_7, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 8] region : (UL : [%32, %c0], LR : [%84, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
                %85 = loom.bufferize_to_tensor %63[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
                %86 = loom.sync ins(%85 : tensor<1x128x64xf16>) outs(%62 : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>
                loom.semaphore_give %63 : memref<1x128x64xf16>
                %87 = linalg.fill ins(%cst : f16) outs(%66 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                %88 = linalg.batch_matmul ins(%35, %86 : tensor<1x1024x128xf16>, tensor<1x128x64xf16>) outs(%87 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                loom.semaphore_give %61 : memref<1x128x64xf16>
                %89 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%88 : tensor<1x1024x64xf16>) outs(%89 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %105 = arith.maximumf %in, %out : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %90 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%53 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.mulf %in_10, %cst_2 : f16
                  %106 = arith.cmpf ogt, %in, %105 : f16
                  %107 = arith.select %106, %in, %105 : f16
                  linalg.yield %107 : f16
                } -> tensor<1x1024x1xf16>
                %92 = loom.broadcast ins(%91 : tensor<1x1024x1xf16>) outs(%71 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x64xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %92 : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>) outs(%66 : tensor<1x1024x64xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.mulf %in, %cst_2 : f16
                  %106 = arith.subf %105, %in_10 : f16
                  %107 = math.exp %106 : f16
                  linalg.yield %107 : f16
                } -> tensor<1x1024x64xf16>
                loom.semaphore_give %70 : memref<1x1024x32xf16>
                %94 = linalg.fill ins(%cst : f16) outs(%56 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%93 : tensor<1x1024x64xf16>) outs(%94 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %105 = arith.addf %in, %out : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %91 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%59 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.subf %in, %in_10 : f16
                  %106 = math.exp %105 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x1xf16>
                %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96, %95 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%arg10 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %105 = arith.mulf %in, %in_10 : f16
                  %106 = arith.addf %105, %in_11 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x1xf16>
                loom.semaphore_give %55 : memref<1x1024x1xf16>
                %c0_8 = arith.constant 0 : index
                %98 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %82, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 8] region : (UL : [%32, %c0], LR : [%84, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
                %99 = loom.bufferize_to_tensor %75[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
                %100 = loom.sync ins(%99 : tensor<1x64x128xf16>) outs(%74 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
                loom.semaphore_give %75 : memref<1x64x128xf16>
                %101 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                %102 = linalg.batch_matmul ins(%93, %100 : tensor<1x1024x64xf16>, tensor<1x64x128xf16>) outs(%101 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                loom.semaphore_give %73 : memref<1x64x128xf16>
                loom.semaphore_give %65 : memref<1x1024x64xf16>
                %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102, %arg11, %96 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x1xf16>) outs(%arg11 : tensor<1x1024x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %105 = arith.mulf %in_10, %in_11 : f16
                  %106 = arith.addf %in, %105 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x128xf16>
                loom.semaphore_give %58 : memref<1x1024x1xf16>
                loom.semaphore_give %49 : memref<1x1024x128xf16>
                %104 = linalg.copy ins(%91 : tensor<1x1024x1xf16>) outs(%arg9 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                loom.semaphore_give %52 : memref<1x1024x1xf16>
                scf.yield %104, %97, %103 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>
              }
              loom.semaphore_give %45 : memref<1x1024x1xf16>
              loom.semaphore_give %29 : memref<1x1024x128xf16>
              %77 = loom.broadcast ins(%76#1 : tensor<1x1024x1xf16>) outs(%69 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x128xf16>
              loom.semaphore_give %41 : memref<1x1024x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76#2, %77 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>) outs(%27 : tensor<1x1024x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %82 = arith.divf %in, %in_6 : f16
                linalg.yield %82 : f16
              } -> tensor<1x1024x128xf16>
              loom.semaphore_give %68 : memref<1x1024x32xf16>
              loom.semaphore_give %37 : memref<1x1024x128xf16>
              %79 = loom.sync ins(%78 : tensor<1x1024x128xf16>) outs(%25 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %26 : memref<1x1024x128xf16>
              %c0_4 = arith.constant 0 : index
              %80 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %22, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%80], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %81 = loom.bufferize_to_memref %79 : tensor<1x1024x128xf16> -> memref<1x1024x128xf16>
              loom.copy %81, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %24 : memref<1x1024x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x4x2_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc4_dim_y_level0_bc8_dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c16 = arith.constant 16 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (4) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c16 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%arg5, %arg6)
              %22 = arith.muli %21, %c1024 : index
              %23 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %25 = loom.init_tensor %24[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %26 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %27 = loom.init_tensor %26[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %28 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %29 = loom.semaphore_take %23 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %30 = loom.init_tensor %29[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %c0_3 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %22, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %32 = arith.muli %arg4, %c4 : index
              %33 = arith.addi %arg5, %32 : index
              loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
              %34 = loom.bufferize_to_tensor %28[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %35 = loom.sync ins(%34 : tensor<1x1024x128xf16>) outs(%30 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %28 : memref<1x1024x128xf16>
              %36 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %37 = loom.semaphore_take %36 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %38 = loom.init_tensor %37[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              %40 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %41 = loom.semaphore_take %40 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %42 = loom.init_tensor %41[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %44 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %45 = loom.semaphore_take %44 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %46 = loom.init_tensor %45[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %48 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %49 = loom.semaphore_take %48 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %50 = loom.init_tensor %49[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %51 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %53 = loom.init_tensor %52[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %54 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %56 = loom.init_tensor %55[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %57 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %58 = loom.semaphore_take %57 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %59 = loom.init_tensor %58[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %60 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
              %61 = loom.semaphore_take %60 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %62 = loom.init_tensor %61[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %63 = loom.semaphore_take %60 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %64 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
              %65 = loom.semaphore_take %64 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
              %66 = loom.init_tensor %65[1, 1024, 64] : memref<1x1024x64xf16> -> tensor<1x1024x64xf16>
              %67 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
              %68 = loom.semaphore_take %67 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %69 = loom.init_tensor %68[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %70 = loom.semaphore_take %67 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %71 = loom.init_tensor %70[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %72 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
              %73 = loom.semaphore_take %72 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %74 = loom.init_tensor %73[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %75 = loom.semaphore_take %72 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %76:3 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %47, %arg10 = %43, %arg11 = %39) -> (tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>) {
                %82 = arith.muli %arg8, %c64 : index
                %c0_6 = arith.constant 0 : index
                %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%20, %c0_6, %82)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%83], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
                %84 = arith.addi %32, %c3 : index
                loom.copy %reinterpret_cast_7, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%32, %c0], LR : [%84, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
                %85 = loom.bufferize_to_tensor %63[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
                %86 = loom.sync ins(%85 : tensor<1x128x64xf16>) outs(%62 : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>
                loom.semaphore_give %63 : memref<1x128x64xf16>
                %87 = linalg.fill ins(%cst : f16) outs(%66 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                %88 = linalg.batch_matmul ins(%35, %86 : tensor<1x1024x128xf16>, tensor<1x128x64xf16>) outs(%87 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                loom.semaphore_give %61 : memref<1x128x64xf16>
                %89 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%88 : tensor<1x1024x64xf16>) outs(%89 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %105 = arith.maximumf %in, %out : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %90 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%53 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.mulf %in_10, %cst_2 : f16
                  %106 = arith.cmpf ogt, %in, %105 : f16
                  %107 = arith.select %106, %in, %105 : f16
                  linalg.yield %107 : f16
                } -> tensor<1x1024x1xf16>
                %92 = loom.broadcast ins(%91 : tensor<1x1024x1xf16>) outs(%71 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x64xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %92 : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>) outs(%66 : tensor<1x1024x64xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.mulf %in, %cst_2 : f16
                  %106 = arith.subf %105, %in_10 : f16
                  %107 = math.exp %106 : f16
                  linalg.yield %107 : f16
                } -> tensor<1x1024x64xf16>
                loom.semaphore_give %70 : memref<1x1024x32xf16>
                %94 = linalg.fill ins(%cst : f16) outs(%56 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%93 : tensor<1x1024x64xf16>) outs(%94 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %105 = arith.addf %in, %out : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %91 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%59 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %105 = arith.subf %in, %in_10 : f16
                  %106 = math.exp %105 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x1xf16>
                %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96, %95 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%arg10 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %105 = arith.mulf %in, %in_10 : f16
                  %106 = arith.addf %105, %in_11 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x1xf16>
                loom.semaphore_give %55 : memref<1x1024x1xf16>
                %c0_8 = arith.constant 0 : index
                %98 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %82, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%98], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%32, %c0], LR : [%84, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
                %99 = loom.bufferize_to_tensor %75[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
                %100 = loom.sync ins(%99 : tensor<1x64x128xf16>) outs(%74 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
                loom.semaphore_give %75 : memref<1x64x128xf16>
                %101 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                %102 = linalg.batch_matmul ins(%93, %100 : tensor<1x1024x64xf16>, tensor<1x64x128xf16>) outs(%101 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                loom.semaphore_give %73 : memref<1x64x128xf16>
                loom.semaphore_give %65 : memref<1x1024x64xf16>
                %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%102, %arg11, %96 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x1xf16>) outs(%arg11 : tensor<1x1024x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %105 = arith.mulf %in_10, %in_11 : f16
                  %106 = arith.addf %in, %105 : f16
                  linalg.yield %106 : f16
                } -> tensor<1x1024x128xf16>
                loom.semaphore_give %58 : memref<1x1024x1xf16>
                loom.semaphore_give %49 : memref<1x1024x128xf16>
                %104 = linalg.copy ins(%91 : tensor<1x1024x1xf16>) outs(%arg9 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                loom.semaphore_give %52 : memref<1x1024x1xf16>
                scf.yield %104, %97, %103 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>
              }
              loom.semaphore_give %45 : memref<1x1024x1xf16>
              loom.semaphore_give %29 : memref<1x1024x128xf16>
              %77 = loom.broadcast ins(%76#1 : tensor<1x1024x1xf16>) outs(%69 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x128xf16>
              loom.semaphore_give %41 : memref<1x1024x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76#2, %77 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>) outs(%27 : tensor<1x1024x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %82 = arith.divf %in, %in_6 : f16
                linalg.yield %82 : f16
              } -> tensor<1x1024x128xf16>
              loom.semaphore_give %68 : memref<1x1024x32xf16>
              loom.semaphore_give %37 : memref<1x1024x128xf16>
              %79 = loom.sync ins(%78 : tensor<1x1024x128xf16>) outs(%25 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %26 : memref<1x1024x128xf16>
              %c0_4 = arith.constant 0 : index
              %80 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%20, %22, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%80], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %81 = loom.bufferize_to_memref %79 : tensor<1x1024x128xf16> -> memref<1x1024x128xf16>
              loom.copy %81, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %24 : memref<1x1024x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8x1_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level1_bc8_dim_y_level0_bc8_dim_x_level1_bc8_dim_y_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c32 step %c1 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %21 = arith.muli %20, %c1024 : index
              %22 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %23 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %24 = loom.init_tensor %23[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %25 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %26 = loom.init_tensor %25[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %27 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %28 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %29 = loom.init_tensor %28[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %c0_3 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %21, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%30], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %31 = arith.muli %arg4, %c8 : index
              %32 = arith.addi %arg5, %31 : index
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%32, %arg6], LR : [%32, %arg6]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
              %33 = loom.bufferize_to_tensor %27[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %34 = loom.sync ins(%33 : tensor<1x1024x128xf16>) outs(%29 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %27 : memref<1x1024x128xf16>
              %35 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %36 = loom.semaphore_take %35 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %37 = loom.init_tensor %36[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              %39 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %40 = loom.semaphore_take %39 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %41 = loom.init_tensor %40[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %42 = linalg.fill ins(%cst_0 : f16) outs(%41 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %43 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %44 = loom.semaphore_take %43 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %45 = loom.init_tensor %44[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %46 = linalg.fill ins(%cst_1 : f16) outs(%45 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
              %47 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
              %48 = loom.semaphore_take %47 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
              %49 = loom.init_tensor %48[1, 1024, 128] : memref<1x1024x128xf16> -> tensor<1x1024x128xf16>
              %50 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %51 = loom.semaphore_take %50 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %52 = loom.init_tensor %51[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %53 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %54 = loom.semaphore_take %53 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %55 = loom.init_tensor %54[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %56 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
              %57 = loom.semaphore_take %56 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
              %58 = loom.init_tensor %57[1, 1024, 1] : memref<1x1024x1xf16> -> tensor<1x1024x1xf16>
              %59 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
              %60 = loom.semaphore_take %59 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %61 = loom.init_tensor %60[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %62 = loom.semaphore_take %59 : memref<1x128x64xf16> -> memref<1x128x64xf16>
              %63 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
              %64 = loom.semaphore_take %63 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
              %65 = loom.init_tensor %64[1, 1024, 64] : memref<1x1024x64xf16> -> tensor<1x1024x64xf16>
              %66 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
              %67 = loom.semaphore_take %66 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %68 = loom.init_tensor %67[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %69 = loom.semaphore_take %66 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
              %70 = loom.init_tensor %69[1, 1024, 32] : memref<1x1024x32xf16> -> tensor<1x1024x32xf16>
              %71 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
              %72 = loom.semaphore_take %71 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %73 = loom.init_tensor %72[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %74 = loom.semaphore_take %71 : memref<1x64x128xf16> -> memref<1x64x128xf16>
              %75:3 = scf.for %arg8 = %c0 to %c64 step %c1 iter_args(%arg9 = %46, %arg10 = %42, %arg11 = %38) -> (tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>) {
                %81 = arith.muli %arg8, %c64 : index
                %c0_6 = arith.constant 0 : index
                %82 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%arg7, %c0_6, %81)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%82], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
                %83 = loom.bufferize_to_tensor %62[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
                %84 = loom.sync ins(%83 : tensor<1x128x64xf16>) outs(%61 : tensor<1x128x64xf16>) -> tensor<1x128x64xf16>
                loom.semaphore_give %62 : memref<1x128x64xf16>
                %85 = linalg.fill ins(%cst : f16) outs(%65 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                %86 = linalg.batch_matmul ins(%34, %84 : tensor<1x1024x128xf16>, tensor<1x128x64xf16>) outs(%85 : tensor<1x1024x64xf16>) -> tensor<1x1024x64xf16>
                loom.semaphore_give %60 : memref<1x128x64xf16>
                %87 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%86 : tensor<1x1024x64xf16>) outs(%87 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %103 = arith.maximumf %in, %out : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x1xf16>
                %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %88 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%52 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %103 = arith.mulf %in_10, %cst_2 : f16
                  %104 = arith.cmpf ogt, %in, %103 : f16
                  %105 = arith.select %104, %in, %103 : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x1xf16>
                %90 = loom.broadcast ins(%89 : tensor<1x1024x1xf16>) outs(%70 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x64xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %90 : tensor<1x1024x64xf16>, tensor<1x1024x64xf16>) outs(%65 : tensor<1x1024x64xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %103 = arith.mulf %in, %cst_2 : f16
                  %104 = arith.subf %103, %in_10 : f16
                  %105 = math.exp %104 : f16
                  linalg.yield %105 : f16
                } -> tensor<1x1024x64xf16>
                loom.semaphore_give %69 : memref<1x1024x32xf16>
                %92 = linalg.fill ins(%cst : f16) outs(%55 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%91 : tensor<1x1024x64xf16>) outs(%92 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %103 = arith.addf %in, %out : f16
                  linalg.yield %103 : f16
                } -> tensor<1x1024x1xf16>
                %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %89 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%58 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %103 = arith.subf %in, %in_10 : f16
                  %104 = math.exp %103 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x1xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %94, %93 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x1xf16>) outs(%arg10 : tensor<1x1024x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %103 = arith.mulf %in, %in_10 : f16
                  %104 = arith.addf %103, %in_11 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x1xf16>
                loom.semaphore_give %54 : memref<1x1024x1xf16>
                %c0_8 = arith.constant 0 : index
                %96 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %81, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%96], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
                %97 = loom.bufferize_to_tensor %74[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
                %98 = loom.sync ins(%97 : tensor<1x64x128xf16>) outs(%73 : tensor<1x64x128xf16>) -> tensor<1x64x128xf16>
                loom.semaphore_give %74 : memref<1x64x128xf16>
                %99 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                %100 = linalg.batch_matmul ins(%91, %98 : tensor<1x1024x64xf16>, tensor<1x64x128xf16>) outs(%99 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
                loom.semaphore_give %72 : memref<1x64x128xf16>
                loom.semaphore_give %64 : memref<1x1024x64xf16>
                %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100, %arg11, %94 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>, tensor<1x1024x1xf16>) outs(%arg11 : tensor<1x1024x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %103 = arith.mulf %in_10, %in_11 : f16
                  %104 = arith.addf %in, %103 : f16
                  linalg.yield %104 : f16
                } -> tensor<1x1024x128xf16>
                loom.semaphore_give %57 : memref<1x1024x1xf16>
                loom.semaphore_give %48 : memref<1x1024x128xf16>
                %102 = linalg.copy ins(%89 : tensor<1x1024x1xf16>) outs(%arg9 : tensor<1x1024x1xf16>) -> tensor<1x1024x1xf16>
                loom.semaphore_give %51 : memref<1x1024x1xf16>
                scf.yield %102, %95, %101 : tensor<1x1024x1xf16>, tensor<1x1024x1xf16>, tensor<1x1024x128xf16>
              }
              loom.semaphore_give %44 : memref<1x1024x1xf16>
              loom.semaphore_give %28 : memref<1x1024x128xf16>
              %76 = loom.broadcast ins(%75#1 : tensor<1x1024x1xf16>) outs(%68 : tensor<1x1024x32xf16>) dim(2) -> tensor<1x1024x128xf16>
              loom.semaphore_give %40 : memref<1x1024x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%75#2, %76 : tensor<1x1024x128xf16>, tensor<1x1024x128xf16>) outs(%26 : tensor<1x1024x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %81 = arith.divf %in, %in_6 : f16
                linalg.yield %81 : f16
              } -> tensor<1x1024x128xf16>
              loom.semaphore_give %67 : memref<1x1024x32xf16>
              loom.semaphore_give %36 : memref<1x1024x128xf16>
              %78 = loom.sync ins(%77 : tensor<1x1024x128xf16>) outs(%24 : tensor<1x1024x128xf16>) -> tensor<1x1024x128xf16>
              loom.semaphore_give %25 : memref<1x1024x128xf16>
              %c0_4 = arith.constant 0 : index
              %79 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %21, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%79], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              %80 = loom.bufferize_to_memref %78 : tensor<1x1024x128xf16> -> memref<1x1024x128xf16>
              loom.copy %80, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%32, %arg6], LR : [%32, %arg6]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %23 : memref<1x1024x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
