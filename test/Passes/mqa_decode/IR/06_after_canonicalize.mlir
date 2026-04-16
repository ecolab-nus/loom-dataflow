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
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_n__tile_b2__tile_n64__tile_s1024(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c16 = arith.constant 16 : index
      %c7 = arith.constant 7 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c2 = arith.constant 2 : index
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          %20 = arith.muli %arg4, %c2 : index
          %21 = loom.alloc [2, 32, 128] on @L1 : memref<2x32x128xf16>
          %22 = loom.semaphore_take %21 : memref<2x32x128xf16> -> memref<2x32x128xf16>
          %23 = loom.init_tensor %22[2, 32, 128] : memref<2x32x128xf16> -> tensor<2x32x128xf16>
          %24 = loom.semaphore_take %21 : memref<2x32x128xf16> -> memref<2x32x128xf16>
          %c0_4 = arith.constant 0 : index
          %c0_5 = arith.constant 0 : index
          %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_4, %c0_5)
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [2, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<2x32x128xf16>
          %26 = loom.bufferize_to_tensor %24[2, 32, 128] : memref<2x32x128xf16> -> tensor<2x32x128xf16>
          %27 = arith.muli %arg5, %c1024 : index
          %28 = arith.addi %27, %c1024 : index
          %29 = loom.alloc [2, 32, 128] on @L1 : memref<2x32x128xf16>
          %30 = loom.semaphore_take %29 : memref<2x32x128xf16> -> memref<2x32x128xf16>
          %31 = loom.init_tensor %30[2, 32, 128] : memref<2x32x128xf16> -> tensor<2x32x128xf16>
          %32 = linalg.fill ins(%cst_0 : f16) outs(%31 : tensor<2x32x128xf16>) -> tensor<2x32x128xf16>
          %33 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %34 = loom.semaphore_take %33 : memref<2x32xf16> -> memref<2x32xf16>
          %35 = loom.init_tensor %34[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %36 = loom.semaphore_take %33 : memref<2x32xf16> -> memref<2x32xf16>
          %37 = loom.init_tensor %36[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %38 = loom.semaphore_take %33 : memref<2x32xf16> -> memref<2x32xf16>
          %39 = loom.init_tensor %38[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %40 = loom.semaphore_take %33 : memref<2x32xf16> -> memref<2x32xf16>
          %41 = loom.init_tensor %40[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %42 = linalg.fill ins(%cst_1 : f16) outs(%41 : tensor<2x32xf16>) -> tensor<2x32xf16>
          %43 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %44 = loom.semaphore_take %43 : memref<2x32xf16> -> memref<2x32xf16>
          %45 = loom.init_tensor %44[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %46 = linalg.fill ins(%cst_2 : f16) outs(%45 : tensor<2x32xf16>) -> tensor<2x32xf16>
          %47 = loom.alloc [2, 32, 128] on @L1 : memref<2x32x128xf16>
          %48 = loom.semaphore_take %47 : memref<2x32x128xf16> -> memref<2x32x128xf16>
          %49 = loom.init_tensor %48[2, 32, 128] : memref<2x32x128xf16> -> tensor<2x32x128xf16>
          %50 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %51 = loom.semaphore_take %50 : memref<2x32xf16> -> memref<2x32xf16>
          %52 = loom.init_tensor %51[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %53 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %54 = loom.semaphore_take %53 : memref<2x32xf16> -> memref<2x32xf16>
          %55 = loom.init_tensor %54[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %56 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %57 = loom.semaphore_take %56 : memref<2x32xf16> -> memref<2x32xf16>
          %58 = loom.init_tensor %57[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %59 = loom.alloc [2, 32, 64] on @L1 : memref<2x32x64xf16>
          %60 = loom.semaphore_take %59 : memref<2x32x64xf16> -> memref<2x32x64xf16>
          %61 = loom.init_tensor %60[2, 32, 64] : memref<2x32x64xf16> -> tensor<2x32x64xf16>
          %62:3 = scf.for %arg6 = %c0 to %c16 step %c1 iter_args(%arg7 = %46, %arg8 = %42, %arg9 = %32) -> (tensor<2x32xf16>, tensor<2x32xf16>, tensor<2x32x128xf16>) {
            %71 = arith.muli %arg6, %c64 : index
            %72 = arith.addi %27, %71 : index
            %73 = arith.addi %72, %c64 : index
            %74 = arith.cmpi ult, %73, %28 : index
            %75 = arith.select %74, %73, %28 : index
            %76 = arith.subi %75, %72 : index
            %77 = loom.alloc [2, 128, %76] on @L1 : memref<?x128x?xf16>
            %78 = loom.semaphore_take %77 : memref<?x128x?xf16> -> memref<?x128x?xf16>
            %c0_6 = arith.constant 0 : index
            %79 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_6, %72)
            %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%79], sizes: [2, 128, %76], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<2x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_7, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<2x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
            %80 = loom.bufferize_to_tensor %78[2, 128, %76] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
            %81 = linalg.fill ins(%cst_0 : f16) outs(%61 : tensor<2x32x64xf16>) -> tensor<2x32x64xf16>
            %cast = tensor.cast %80 : tensor<?x128x?xf16> to tensor<2x128x64xf16>
            %82 = linalg.batch_matmul ins(%26, %cast : tensor<2x32x128xf16>, tensor<2x128x64xf16>) outs(%81 : tensor<2x32x64xf16>) -> tensor<2x32x64xf16>
            loom.semaphore_give %78 : memref<?x128x?xf16>
            %83 = linalg.fill ins(%cst_2 : f16) outs(%52 : tensor<2x32xf16>) -> tensor<2x32xf16>
            %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%82 : tensor<2x32x64xf16>) outs(%83 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %99 = arith.maximumf %in, %out : f16
              linalg.yield %99 : f16
            } -> tensor<2x32xf16>
            %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %84 : tensor<2x32xf16>, tensor<2x32xf16>) outs(%52 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %in_11: f16, %out: f16):
              %99 = arith.mulf %in_11, %cst_3 : f16
              %100 = arith.cmpf ogt, %in, %99 : f16
              %101 = arith.select %100, %in, %99 : f16
              linalg.yield %101 : f16
            } -> tensor<2x32xf16>
            %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %85 : tensor<2x32x64xf16>, tensor<2x32xf16>) outs(%61 : tensor<2x32x64xf16>) {
            ^bb0(%in: f16, %in_11: f16, %out: f16):
              %99 = arith.mulf %in, %cst_3 : f16
              %100 = arith.subf %99, %in_11 : f16
              %101 = math.powf %cst, %100 : f16
              linalg.yield %101 : f16
            } -> tensor<2x32x64xf16>
            %87 = linalg.fill ins(%cst_0 : f16) outs(%55 : tensor<2x32xf16>) -> tensor<2x32xf16>
            %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%86 : tensor<2x32x64xf16>) outs(%87 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %99 = arith.addf %in, %out : f16
              linalg.yield %99 : f16
            } -> tensor<2x32xf16>
            %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %85 : tensor<2x32xf16>, tensor<2x32xf16>) outs(%58 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %in_11: f16, %out: f16):
              %99 = arith.subf %in, %in_11 : f16
              %100 = math.powf %cst, %99 : f16
              linalg.yield %100 : f16
            } -> tensor<2x32xf16>
            %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %89, %88 : tensor<2x32xf16>, tensor<2x32xf16>, tensor<2x32xf16>) outs(%arg8 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %in_11: f16, %in_12: f16, %out: f16):
              %99 = arith.mulf %in, %in_11 : f16
              %100 = arith.addf %99, %in_12 : f16
              linalg.yield %100 : f16
            } -> tensor<2x32xf16>
            loom.semaphore_give %54 : memref<2x32xf16>
            %91 = loom.alloc [2, %76, 128] on @L1 : memref<?x?x128xf16>
            %92 = loom.semaphore_take %91 : memref<?x?x128xf16> -> memref<?x?x128xf16>
            %c0_8 = arith.constant 0 : index
            %93 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %72, %c0_8)
            %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%93], sizes: [2, %76, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<2x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<2x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
            %94 = loom.bufferize_to_tensor %92[2, %76, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
            %95 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<2x32x128xf16>) -> tensor<2x32x128xf16>
            %cast_10 = tensor.cast %94 : tensor<?x?x128xf16> to tensor<2x64x128xf16>
            %96 = linalg.batch_matmul ins(%86, %cast_10 : tensor<2x32x64xf16>, tensor<2x64x128xf16>) outs(%95 : tensor<2x32x128xf16>) -> tensor<2x32x128xf16>
            loom.semaphore_give %92 : memref<?x?x128xf16>
            loom.semaphore_give %60 : memref<2x32x64xf16>
            %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%96, %arg9, %89 : tensor<2x32x128xf16>, tensor<2x32x128xf16>, tensor<2x32xf16>) outs(%arg9 : tensor<2x32x128xf16>) {
            ^bb0(%in: f16, %in_11: f16, %in_12: f16, %out: f16):
              %99 = arith.mulf %in_11, %in_12 : f16
              %100 = arith.addf %in, %99 : f16
              linalg.yield %100 : f16
            } -> tensor<2x32x128xf16>
            loom.semaphore_give %57 : memref<2x32xf16>
            loom.semaphore_give %48 : memref<2x32x128xf16>
            %98 = linalg.copy ins(%85 : tensor<2x32xf16>) outs(%arg7 : tensor<2x32xf16>) -> tensor<2x32xf16>
            loom.semaphore_give %51 : memref<2x32xf16>
            scf.yield %98, %90, %97 : tensor<2x32xf16>, tensor<2x32xf16>, tensor<2x32x128xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %24 : memref<2x32x128xf16>
          %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%62#1, %62#0 : tensor<2x32xf16>, tensor<2x32xf16>) outs(%39 : tensor<2x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %71 = math.log2 %in : f16
            %72 = arith.addf %71, %in_6 : f16
            linalg.yield %72 : f16
          } -> tensor<2x32xf16>
          loom.semaphore_give %40 : memref<2x32xf16>
          loom.semaphore_give %44 : memref<2x32xf16>
          %64 = loom.alloc [8, 2, 32] on @L1 : memref<8x2x32xf16>
          %65 = loom.semaphore_take %64 : memref<8x2x32xf16> -> memref<8x2x32xf16>
          %66 = loom.init_tensor %65[8, 2, 32] : memref<8x2x32xf16> -> tensor<8x2x32xf16>
          %67 = loom.alloc [8, 2, 32, 128] on @L1 : memref<8x2x32x128xf16>
          %68 = loom.semaphore_take %67 : memref<8x2x32x128xf16> -> memref<8x2x32x128xf16>
          %69 = loom.init_tensor %68[8, 2, 32, 128] : memref<8x2x32x128xf16> -> tensor<8x2x32x128xf16>
          %70 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %70 {
            %71 = loom.gather ins(%63 : tensor<2x32xf16>) outs(%66 : tensor<8x2x32xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x2x32xf16>
            loom.semaphore_give %38 : memref<2x32xf16>
            %72 = linalg.fill ins(%cst_2 : f16) outs(%37 : tensor<2x32xf16>) -> tensor<2x32xf16>
            %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%71 : tensor<8x2x32xf16>) outs(%72 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.maximumf %in, %out : f16
              linalg.yield %83 : f16
            } -> tensor<2x32xf16>
            %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71, %73 : tensor<8x2x32xf16>, tensor<2x32xf16>) outs(%66 : tensor<8x2x32xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %83 = arith.subf %in, %in_9 : f16
              %84 = math.powf %cst, %83 : f16
              linalg.yield %84 : f16
            } -> tensor<8x2x32xf16>
            loom.semaphore_give %36 : memref<2x32xf16>
            %75 = linalg.fill ins(%cst_0 : f16) outs(%35 : tensor<2x32xf16>) -> tensor<2x32xf16>
            %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<8x2x32xf16>) outs(%75 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %83 = arith.addf %in, %out : f16
              linalg.yield %83 : f16
            } -> tensor<2x32xf16>
            %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %76 : tensor<8x2x32xf16>, tensor<2x32xf16>) outs(%66 : tensor<8x2x32xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %83 = arith.divf %in, %in_9 : f16
              linalg.yield %83 : f16
            } -> tensor<8x2x32xf16>
            loom.semaphore_give %34 : memref<2x32xf16>
            %78 = loom.gather ins(%62#2 : tensor<2x32x128xf16>) outs(%69 : tensor<8x2x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x2x32x128xf16>
            %79 = linalg.fill ins(%cst_0 : f16) outs(%23 : tensor<2x32x128xf16>) -> tensor<2x32x128xf16>
            %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%78, %77 : tensor<8x2x32x128xf16>, tensor<8x2x32xf16>) outs(%79 : tensor<2x32x128xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %83 = arith.mulf %in, %in_9 : f16
              %84 = arith.addf %83, %out : f16
              linalg.yield %84 : f16
            } -> tensor<2x32x128xf16>
            loom.semaphore_give %68 : memref<8x2x32x128xf16>
            loom.semaphore_give %65 : memref<8x2x32xf16>
            %c0_6 = arith.constant 0 : index
            %c0_7 = arith.constant 0 : index
            %81 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_6, %c0_7)
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%81], sizes: [2, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            %82 = loom.bufferize_to_memref %80 : tensor<2x32x128xf16> -> memref<2x32x128xf16>
            loom.copy %82, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<2x32x128xf16> to memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %22 : memref<2x32x128xf16>
          }
          loom.semaphore_give %30 : memref<2x32x128xf16>
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
