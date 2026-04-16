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
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %20 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %21 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %23 = arith.ceildivui %c16, %20 : index
      %24 = arith.ceildivui %c8192, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %25 = arith.ceildivui %23, %c8 : index
          scf.for %arg6 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %29 = arith.muli %27, %20 : index
              %30 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %32 = loom.init_tensor %31[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %33 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %34 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %34, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
              %35 = loom.bufferize_to_tensor %33[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %36 = arith.muli %28, %21 : index
              %37 = arith.ceildivui %21, %22 : index
              %38 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %39 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %40 = loom.init_tensor %39[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
              %42 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %43 = loom.semaphore_take %42 : memref<?x32xf16> -> memref<?x32xf16>
              %44 = loom.init_tensor %43[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %45 = loom.semaphore_take %42 : memref<?x32xf16> -> memref<?x32xf16>
              %46 = loom.init_tensor %45[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %47 = loom.semaphore_take %42 : memref<?x32xf16> -> memref<?x32xf16>
              %48 = loom.init_tensor %47[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %49 = loom.semaphore_take %42 : memref<?x32xf16> -> memref<?x32xf16>
              %50 = loom.init_tensor %49[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32xf16>) -> tensor<?x32xf16>
              %52 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %53 = loom.semaphore_take %52 : memref<?x32xf16> -> memref<?x32xf16>
              %54 = loom.init_tensor %53[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %55 = linalg.fill ins(%cst_2 : f16) outs(%54 : tensor<?x32xf16>) -> tensor<?x32xf16>
              %56 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %57 = loom.semaphore_take %56 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %58 = loom.init_tensor %57[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %59 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %60 = loom.semaphore_take %59 : memref<?x32xf16> -> memref<?x32xf16>
              %61 = loom.init_tensor %60[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %62 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %63 = loom.semaphore_take %62 : memref<?x32xf16> -> memref<?x32xf16>
              %64 = loom.init_tensor %63[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %65 = loom.alloc [%20, 32] on @L1 : memref<?x32xf16>
              %66 = loom.semaphore_take %65 : memref<?x32xf16> -> memref<?x32xf16>
              %67 = loom.init_tensor %66[%20, 32] : memref<?x32xf16> -> tensor<?x32xf16>
              %68 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %69 = loom.semaphore_take %68 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %70 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
              %71 = loom.semaphore_take %70 : memref<?x32x?xf16> -> memref<?x32x?xf16>
              %72 = loom.init_tensor %71[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
              %73 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %74 = loom.semaphore_take %73 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %75:3 = scf.for %arg8 = %c0 to %37 step %c1 iter_args(%arg9 = %55, %arg10 = %51, %arg11 = %41) -> (tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>) {
                %84 = arith.muli %arg8, %22 : index
                %85 = arith.addi %36, %84 : index
                %86 = loom.subview %arg0[%29, 0, %85] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %86, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                %87 = loom.bufferize_to_tensor %69[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%72 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                %89 = linalg.batch_matmul ins(%35, %87 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%88 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                loom.semaphore_give %69 : memref<?x128x?xf16>
                %90 = linalg.fill ins(%cst_2 : f16) outs(%61 : tensor<?x32xf16>) -> tensor<?x32xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%89 : tensor<?x32x?xf16>) outs(%90 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %104 = arith.maximumf %in, %out : f16
                  linalg.yield %104 : f16
                } -> tensor<?x32xf16>
                %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %91 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%61 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %104 = arith.mulf %in_4, %cst_3 : f16
                  %105 = arith.cmpf ogt, %in, %104 : f16
                  %106 = arith.select %105, %in, %104 : f16
                  linalg.yield %106 : f16
                } -> tensor<?x32xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %92 : tensor<?x32x?xf16>, tensor<?x32xf16>) outs(%72 : tensor<?x32x?xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %104 = arith.mulf %in, %cst_3 : f16
                  %105 = arith.subf %104, %in_4 : f16
                  %106 = math.powf %cst, %105 : f16
                  linalg.yield %106 : f16
                } -> tensor<?x32x?xf16>
                %94 = linalg.fill ins(%cst_0 : f16) outs(%64 : tensor<?x32xf16>) -> tensor<?x32xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%93 : tensor<?x32x?xf16>) outs(%94 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %104 = arith.addf %in, %out : f16
                  linalg.yield %104 : f16
                } -> tensor<?x32xf16>
                %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %92 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%67 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %104 = arith.subf %in, %in_4 : f16
                  %105 = math.powf %cst, %104 : f16
                  linalg.yield %105 : f16
                } -> tensor<?x32xf16>
                %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg10, %96, %95 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32xf16>) outs(%arg10 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %104 = arith.mulf %in, %in_4 : f16
                  %105 = arith.addf %104, %in_5 : f16
                  linalg.yield %105 : f16
                } -> tensor<?x32xf16>
                loom.semaphore_give %63 : memref<?x32xf16>
                %98 = loom.subview %arg1[%29, %85, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %98, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %99 = loom.bufferize_to_tensor %74[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %100 = linalg.fill ins(%cst_0 : f16) outs(%58 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %101 = linalg.batch_matmul ins(%93, %99 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%100 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                loom.semaphore_give %74 : memref<?x?x128xf16>
                loom.semaphore_give %71 : memref<?x32x?xf16>
                %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101, %arg11, %96 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32xf16>) outs(%arg11 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
                  %104 = arith.mulf %in_4, %in_5 : f16
                  %105 = arith.addf %in, %104 : f16
                  linalg.yield %105 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %66 : memref<?x32xf16>
                loom.semaphore_give %57 : memref<?x32x128xf16>
                %103 = linalg.copy ins(%92 : tensor<?x32xf16>) outs(%arg9 : tensor<?x32xf16>) -> tensor<?x32xf16>
                loom.semaphore_give %60 : memref<?x32xf16>
                scf.yield %103, %97, %102 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %33 : memref<?x32x128xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75#1, %75#0 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%48 : tensor<?x32xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %84 = math.log2 %in : f16
                %85 = arith.addf %84, %in_4 : f16
                linalg.yield %85 : f16
              } -> tensor<?x32xf16>
              loom.semaphore_give %49 : memref<?x32xf16>
              loom.semaphore_give %53 : memref<?x32xf16>
              %77 = loom.alloc [%24, %20, 32] on @L1 : memref<?x?x32xf16>
              %78 = loom.semaphore_take %77 : memref<?x?x32xf16> -> memref<?x?x32xf16>
              %79 = loom.init_tensor %78[%24, %20, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
              %80 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
              %81 = loom.semaphore_take %80 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
              %82 = loom.init_tensor %81[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
              %83 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %83 {
                %84 = loom.gather ins(%76 : tensor<?x32xf16>) outs(%79 : tensor<?x?x32xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<?x?x32xf16>
                loom.semaphore_give %47 : memref<?x32xf16>
                %85 = linalg.fill ins(%cst_2 : f16) outs(%46 : tensor<?x32xf16>) -> tensor<?x32xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%84 : tensor<?x?x32xf16>) outs(%85 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %96 = arith.maximumf %in, %out : f16
                  linalg.yield %96 : f16
                } -> tensor<?x32xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %86 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%79 : tensor<?x?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %96 = arith.subf %in, %in_4 : f16
                  %97 = math.powf %cst, %96 : f16
                  linalg.yield %97 : f16
                } -> tensor<?x?x32xf16>
                loom.semaphore_give %45 : memref<?x32xf16>
                %88 = linalg.fill ins(%cst_0 : f16) outs(%44 : tensor<?x32xf16>) -> tensor<?x32xf16>
                %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%87 : tensor<?x?x32xf16>) outs(%88 : tensor<?x32xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %96 = arith.addf %in, %out : f16
                  linalg.yield %96 : f16
                } -> tensor<?x32xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %89 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%79 : tensor<?x?x32xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %96 = arith.divf %in, %in_4 : f16
                  linalg.yield %96 : f16
                } -> tensor<?x?x32xf16>
                loom.semaphore_give %43 : memref<?x32xf16>
                %91 = loom.gather ins(%75#2 : tensor<?x32x128xf16>) outs(%82 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<?x?x32x128xf16>
                %92 = linalg.fill ins(%cst_0 : f16) outs(%32 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%91, %90 : tensor<?x?x32x128xf16>, tensor<?x?x32xf16>) outs(%92 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %96 = arith.mulf %in, %in_4 : f16
                  %97 = arith.addf %96, %out : f16
                  linalg.yield %97 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %81 : memref<?x?x32x128xf16>
                loom.semaphore_give %78 : memref<?x?x32xf16>
                %94 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %95 = loom.bufferize_to_memref %93 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                loom.copy %95, %94 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %31 : memref<?x32x128xf16>
              }
              loom.semaphore_give %39 : memref<?x32x128xf16>
            } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
