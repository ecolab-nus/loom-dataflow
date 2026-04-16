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
          %28 = loom.alloc [2, 32, 128] on @L1 : memref<2x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<2x32x128xf16> -> memref<2x32x128xf16>
          %30 = loom.init_tensor %29[2, 32, 128] : memref<2x32x128xf16> -> tensor<2x32x128xf16>
          %31 = linalg.fill ins(%cst_0 : f16) outs(%30 : tensor<2x32x128xf16>) -> tensor<2x32x128xf16>
          %32 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %33 = loom.semaphore_take %32 : memref<2x32xf16> -> memref<2x32xf16>
          %34 = loom.init_tensor %33[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %35 = loom.semaphore_take %32 : memref<2x32xf16> -> memref<2x32xf16>
          %36 = loom.init_tensor %35[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %37 = loom.semaphore_take %32 : memref<2x32xf16> -> memref<2x32xf16>
          %38 = loom.init_tensor %37[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %39 = loom.semaphore_take %32 : memref<2x32xf16> -> memref<2x32xf16>
          %40 = loom.init_tensor %39[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %41 = linalg.fill ins(%cst_1 : f16) outs(%40 : tensor<2x32xf16>) -> tensor<2x32xf16>
          %42 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %43 = loom.semaphore_take %42 : memref<2x32xf16> -> memref<2x32xf16>
          %44 = loom.init_tensor %43[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %45 = linalg.fill ins(%cst_2 : f16) outs(%44 : tensor<2x32xf16>) -> tensor<2x32xf16>
          %46 = loom.alloc [2, 32, 128] on @L1 : memref<2x32x128xf16>
          %47 = loom.semaphore_take %46 : memref<2x32x128xf16> -> memref<2x32x128xf16>
          %48 = loom.init_tensor %47[2, 32, 128] : memref<2x32x128xf16> -> tensor<2x32x128xf16>
          %49 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %50 = loom.semaphore_take %49 : memref<2x32xf16> -> memref<2x32xf16>
          %51 = loom.init_tensor %50[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %52 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %53 = loom.semaphore_take %52 : memref<2x32xf16> -> memref<2x32xf16>
          %54 = loom.init_tensor %53[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %55 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
          %56 = loom.semaphore_take %55 : memref<2x32xf16> -> memref<2x32xf16>
          %57 = loom.init_tensor %56[2, 32] : memref<2x32xf16> -> tensor<2x32xf16>
          %58 = loom.alloc [2, 128, 64] on @L1 : memref<2x128x64xf16>
          %59 = loom.semaphore_take %58 : memref<2x128x64xf16> -> memref<2x128x64xf16>
          %60 = loom.alloc [2, 32, 64] on @L1 : memref<2x32x64xf16>
          %61 = loom.semaphore_take %60 : memref<2x32x64xf16> -> memref<2x32x64xf16>
          %62 = loom.init_tensor %61[2, 32, 64] : memref<2x32x64xf16> -> tensor<2x32x64xf16>
          %63 = loom.alloc [2, 64, 128] on @L1 : memref<2x64x128xf16>
          %64 = loom.semaphore_take %63 : memref<2x64x128xf16> -> memref<2x64x128xf16>
          %65:3 = scf.for %arg6 = %c0 to %c16 step %c1 iter_args(%arg7 = %45, %arg8 = %41, %arg9 = %31) -> (tensor<2x32xf16>, tensor<2x32xf16>, tensor<2x32x128xf16>) {
            %74 = arith.muli %arg6, %c64 : index
            %75 = arith.addi %27, %74 : index
            %c0_6 = arith.constant 0 : index
            %76 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_6, %75)
            %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%76], sizes: [2, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<2x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_7, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<2x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<2x128x64xf16>
            %77 = loom.bufferize_to_tensor %59[2, 128, 64] : memref<2x128x64xf16> -> tensor<2x128x64xf16>
            %78 = linalg.fill ins(%cst_0 : f16) outs(%62 : tensor<2x32x64xf16>) -> tensor<2x32x64xf16>
            %79 = linalg.batch_matmul ins(%26, %77 : tensor<2x32x128xf16>, tensor<2x128x64xf16>) outs(%78 : tensor<2x32x64xf16>) -> tensor<2x32x64xf16>
            loom.semaphore_give %59 : memref<2x128x64xf16>
            %80 = linalg.fill ins(%cst_2 : f16) outs(%51 : tensor<2x32xf16>) -> tensor<2x32xf16>
            %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<2x32x64xf16>) outs(%80 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %94 = arith.maximumf %in, %out : f16
              linalg.yield %94 : f16
            } -> tensor<2x32xf16>
            %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %81 : tensor<2x32xf16>, tensor<2x32xf16>) outs(%51 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %94 = arith.mulf %in_10, %cst_3 : f16
              %95 = arith.cmpf ogt, %in, %94 : f16
              %96 = arith.select %95, %in, %94 : f16
              linalg.yield %96 : f16
            } -> tensor<2x32xf16>
            %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %82 : tensor<2x32x64xf16>, tensor<2x32xf16>) outs(%62 : tensor<2x32x64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %94 = arith.mulf %in, %cst_3 : f16
              %95 = arith.subf %94, %in_10 : f16
              %96 = math.powf %cst, %95 : f16
              linalg.yield %96 : f16
            } -> tensor<2x32x64xf16>
            %84 = linalg.fill ins(%cst_0 : f16) outs(%54 : tensor<2x32xf16>) -> tensor<2x32xf16>
            %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%83 : tensor<2x32x64xf16>) outs(%84 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %94 = arith.addf %in, %out : f16
              linalg.yield %94 : f16
            } -> tensor<2x32xf16>
            %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %82 : tensor<2x32xf16>, tensor<2x32xf16>) outs(%57 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %94 = arith.subf %in, %in_10 : f16
              %95 = math.powf %cst, %94 : f16
              linalg.yield %95 : f16
            } -> tensor<2x32xf16>
            %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %86, %85 : tensor<2x32xf16>, tensor<2x32xf16>, tensor<2x32xf16>) outs(%arg8 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
              %94 = arith.mulf %in, %in_10 : f16
              %95 = arith.addf %94, %in_11 : f16
              linalg.yield %95 : f16
            } -> tensor<2x32xf16>
            loom.semaphore_give %53 : memref<2x32xf16>
            %c0_8 = arith.constant 0 : index
            %88 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %75, %c0_8)
            %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%88], sizes: [2, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<2x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_9, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<2x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<2x64x128xf16>
            %89 = loom.bufferize_to_tensor %64[2, 64, 128] : memref<2x64x128xf16> -> tensor<2x64x128xf16>
            %90 = linalg.fill ins(%cst_0 : f16) outs(%48 : tensor<2x32x128xf16>) -> tensor<2x32x128xf16>
            %91 = linalg.batch_matmul ins(%83, %89 : tensor<2x32x64xf16>, tensor<2x64x128xf16>) outs(%90 : tensor<2x32x128xf16>) -> tensor<2x32x128xf16>
            loom.semaphore_give %64 : memref<2x64x128xf16>
            loom.semaphore_give %61 : memref<2x32x64xf16>
            %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %arg9, %86 : tensor<2x32x128xf16>, tensor<2x32x128xf16>, tensor<2x32xf16>) outs(%arg9 : tensor<2x32x128xf16>) {
            ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
              %94 = arith.mulf %in_10, %in_11 : f16
              %95 = arith.addf %in, %94 : f16
              linalg.yield %95 : f16
            } -> tensor<2x32x128xf16>
            loom.semaphore_give %56 : memref<2x32xf16>
            loom.semaphore_give %47 : memref<2x32x128xf16>
            %93 = linalg.copy ins(%82 : tensor<2x32xf16>) outs(%arg7 : tensor<2x32xf16>) -> tensor<2x32xf16>
            loom.semaphore_give %50 : memref<2x32xf16>
            scf.yield %93, %87, %92 : tensor<2x32xf16>, tensor<2x32xf16>, tensor<2x32x128xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %24 : memref<2x32x128xf16>
          %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65#1, %65#0 : tensor<2x32xf16>, tensor<2x32xf16>) outs(%38 : tensor<2x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %74 = math.log2 %in : f16
            %75 = arith.addf %74, %in_6 : f16
            linalg.yield %75 : f16
          } -> tensor<2x32xf16>
          loom.semaphore_give %39 : memref<2x32xf16>
          loom.semaphore_give %43 : memref<2x32xf16>
          %67 = loom.alloc [8, 2, 32] on @L1 : memref<8x2x32xf16>
          %68 = loom.semaphore_take %67 : memref<8x2x32xf16> -> memref<8x2x32xf16>
          %69 = loom.init_tensor %68[8, 2, 32] : memref<8x2x32xf16> -> tensor<8x2x32xf16>
          %70 = loom.alloc [8, 2, 32, 128] on @L1 : memref<8x2x32x128xf16>
          %71 = loom.semaphore_take %70 : memref<8x2x32x128xf16> -> memref<8x2x32x128xf16>
          %72 = loom.init_tensor %71[8, 2, 32, 128] : memref<8x2x32x128xf16> -> tensor<8x2x32x128xf16>
          %73 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %73 {
            %74 = loom.gather ins(%66 : tensor<2x32xf16>) outs(%69 : tensor<8x2x32xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x2x32xf16>
            loom.semaphore_give %37 : memref<2x32xf16>
            %75 = linalg.fill ins(%cst_2 : f16) outs(%36 : tensor<2x32xf16>) -> tensor<2x32xf16>
            %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%74 : tensor<8x2x32xf16>) outs(%75 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %86 = arith.maximumf %in, %out : f16
              linalg.yield %86 : f16
            } -> tensor<2x32xf16>
            %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %76 : tensor<8x2x32xf16>, tensor<2x32xf16>) outs(%69 : tensor<8x2x32xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %86 = arith.subf %in, %in_9 : f16
              %87 = math.powf %cst, %86 : f16
              linalg.yield %87 : f16
            } -> tensor<8x2x32xf16>
            loom.semaphore_give %35 : memref<2x32xf16>
            %78 = linalg.fill ins(%cst_0 : f16) outs(%34 : tensor<2x32xf16>) -> tensor<2x32xf16>
            %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%77 : tensor<8x2x32xf16>) outs(%78 : tensor<2x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %86 = arith.addf %in, %out : f16
              linalg.yield %86 : f16
            } -> tensor<2x32xf16>
            %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %79 : tensor<8x2x32xf16>, tensor<2x32xf16>) outs(%69 : tensor<8x2x32xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %86 = arith.divf %in, %in_9 : f16
              linalg.yield %86 : f16
            } -> tensor<8x2x32xf16>
            loom.semaphore_give %33 : memref<2x32xf16>
            %81 = loom.gather ins(%65#2 : tensor<2x32x128xf16>) outs(%72 : tensor<8x2x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x2x32x128xf16>
            %82 = linalg.fill ins(%cst_0 : f16) outs(%23 : tensor<2x32x128xf16>) -> tensor<2x32x128xf16>
            %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%81, %80 : tensor<8x2x32x128xf16>, tensor<8x2x32xf16>) outs(%82 : tensor<2x32x128xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %86 = arith.mulf %in, %in_9 : f16
              %87 = arith.addf %86, %out : f16
              linalg.yield %87 : f16
            } -> tensor<2x32x128xf16>
            loom.semaphore_give %71 : memref<8x2x32x128xf16>
            loom.semaphore_give %68 : memref<8x2x32xf16>
            %c0_6 = arith.constant 0 : index
            %c0_7 = arith.constant 0 : index
            %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_6, %c0_7)
            %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%84], sizes: [2, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            %85 = loom.bufferize_to_memref %83 : tensor<2x32x128xf16> -> memref<2x32x128xf16>
            loom.copy %85, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<2x32x128xf16> to memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %22 : memref<2x32x128xf16>
          }
          loom.semaphore_give %29 : memref<2x32x128xf16>
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
