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
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_dim_x_level0_bc8__tile_b1__tile_n64__tile_s1024(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
            %21 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %22 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %23 = loom.init_tensor %22[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %24 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %c0_4 = arith.constant 0 : index
            %c0_5 = arith.constant 0 : index
            %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_4, %c0_5)
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
            %26 = loom.bufferize_to_tensor %24[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %27 = arith.muli %arg5, %c1024 : index
            %28 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %29 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %30 = loom.init_tensor %29[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %31 = linalg.fill ins(%cst_0 : f16) outs(%30 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
            %32 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %33 = loom.semaphore_take %32 : memref<1x32xf16> -> memref<1x32xf16>
            %34 = loom.init_tensor %33[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %35 = loom.semaphore_take %32 : memref<1x32xf16> -> memref<1x32xf16>
            %36 = loom.init_tensor %35[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %37 = loom.semaphore_take %32 : memref<1x32xf16> -> memref<1x32xf16>
            %38 = loom.init_tensor %37[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %39 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<1x32xf16>) -> tensor<1x32xf16>
            %40 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %41 = loom.semaphore_take %40 : memref<1x32xf16> -> memref<1x32xf16>
            %42 = loom.init_tensor %41[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %43 = linalg.fill ins(%cst_2 : f16) outs(%42 : tensor<1x32xf16>) -> tensor<1x32xf16>
            %44 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %45 = loom.semaphore_take %44 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %46 = loom.init_tensor %45[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %47 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %48 = loom.semaphore_take %47 : memref<1x32xf16> -> memref<1x32xf16>
            %49 = loom.init_tensor %48[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %50 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %51 = loom.semaphore_take %50 : memref<1x32xf16> -> memref<1x32xf16>
            %52 = loom.init_tensor %51[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %53 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %54 = loom.semaphore_take %53 : memref<1x32xf16> -> memref<1x32xf16>
            %55 = loom.init_tensor %54[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %56 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
            %57 = loom.semaphore_take %56 : memref<1x128x64xf16> -> memref<1x128x64xf16>
            %58 = loom.alloc [1, 32, 64] on @L1 : memref<1x32x64xf16>
            %59 = loom.semaphore_take %58 : memref<1x32x64xf16> -> memref<1x32x64xf16>
            %60 = loom.init_tensor %59[1, 32, 64] : memref<1x32x64xf16> -> tensor<1x32x64xf16>
            %61 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
            %62 = loom.semaphore_take %61 : memref<1x64x128xf16> -> memref<1x64x128xf16>
            %63:3 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %43, %arg9 = %39, %arg10 = %31) -> (tensor<1x32xf16>, tensor<1x32xf16>, tensor<1x32x128xf16>) {
              %74 = arith.muli %arg7, %c64 : index
              %75 = arith.addi %27, %74 : index
              %c0_6 = arith.constant 0 : index
              %76 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_6, %75)
              %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%76], sizes: [1, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
              loom.copy %reinterpret_cast_7, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x64xf16>
              %77 = loom.bufferize_to_tensor %57[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %78 = linalg.fill ins(%cst_0 : f16) outs(%60 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              %79 = linalg.batch_matmul ins(%26, %77 : tensor<1x32x128xf16>, tensor<1x128x64xf16>) outs(%78 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              loom.semaphore_give %57 : memref<1x128x64xf16>
              %80 = linalg.fill ins(%cst_2 : f16) outs(%49 : tensor<1x32xf16>) -> tensor<1x32xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x32x64xf16>) outs(%80 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %94 = arith.maximumf %in, %out : f16
                linalg.yield %94 : f16
              } -> tensor<1x32xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %81 : tensor<1x32xf16>, tensor<1x32xf16>) outs(%49 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %in_10: f16, %out: f16):
                %94 = arith.mulf %in_10, %cst_3 : f16
                %95 = arith.cmpf ogt, %in, %94 : f16
                %96 = arith.select %95, %in, %94 : f16
                linalg.yield %96 : f16
              } -> tensor<1x32xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %82 : tensor<1x32x64xf16>, tensor<1x32xf16>) outs(%60 : tensor<1x32x64xf16>) {
              ^bb0(%in: f16, %in_10: f16, %out: f16):
                %94 = arith.mulf %in, %cst_3 : f16
                %95 = arith.subf %94, %in_10 : f16
                %96 = math.powf %cst, %95 : f16
                linalg.yield %96 : f16
              } -> tensor<1x32x64xf16>
              %84 = linalg.fill ins(%cst_0 : f16) outs(%52 : tensor<1x32xf16>) -> tensor<1x32xf16>
              %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%83 : tensor<1x32x64xf16>) outs(%84 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %94 = arith.addf %in, %out : f16
                linalg.yield %94 : f16
              } -> tensor<1x32xf16>
              %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %82 : tensor<1x32xf16>, tensor<1x32xf16>) outs(%55 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %in_10: f16, %out: f16):
                %94 = arith.subf %in, %in_10 : f16
                %95 = math.powf %cst, %94 : f16
                linalg.yield %95 : f16
              } -> tensor<1x32xf16>
              %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %86, %85 : tensor<1x32xf16>, tensor<1x32xf16>, tensor<1x32xf16>) outs(%arg9 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                %94 = arith.mulf %in, %in_10 : f16
                %95 = arith.addf %94, %in_11 : f16
                linalg.yield %95 : f16
              } -> tensor<1x32xf16>
              loom.semaphore_give %51 : memref<1x32xf16>
              %c0_8 = arith.constant 0 : index
              %88 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %75, %c0_8)
              %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%88], sizes: [1, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_9, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x64x128xf16>
              %89 = loom.bufferize_to_tensor %62[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %90 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %91 = linalg.batch_matmul ins(%83, %89 : tensor<1x32x64xf16>, tensor<1x64x128xf16>) outs(%90 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %62 : memref<1x64x128xf16>
              loom.semaphore_give %59 : memref<1x32x64xf16>
              %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %arg10, %86 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32xf16>) outs(%arg10 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                %94 = arith.mulf %in_10, %in_11 : f16
                %95 = arith.addf %in, %94 : f16
                linalg.yield %95 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %54 : memref<1x32xf16>
              loom.semaphore_give %45 : memref<1x32x128xf16>
              %93 = linalg.copy ins(%82 : tensor<1x32xf16>) outs(%arg8 : tensor<1x32xf16>) -> tensor<1x32xf16>
              loom.semaphore_give %48 : memref<1x32xf16>
              scf.yield %93, %87, %92 : tensor<1x32xf16>, tensor<1x32xf16>, tensor<1x32x128xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %24 : memref<1x32x128xf16>
            %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63#1, %63#0 : tensor<1x32xf16>, tensor<1x32xf16>) outs(%36 : tensor<1x32xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %74 = math.log2 %in : f16
              %75 = arith.addf %74, %in_6 : f16
              linalg.yield %75 : f16
            } -> tensor<1x32xf16>
            loom.semaphore_give %37 : memref<1x32xf16>
            loom.semaphore_give %41 : memref<1x32xf16>
            %65 = loom.alloc [8, 1, 32] on @L1 : memref<8x1x32xf16>
            %66 = loom.semaphore_take %65 : memref<8x1x32xf16> -> memref<8x1x32xf16>
            %67 = loom.init_tensor %66[8, 1, 32] : memref<8x1x32xf16> -> tensor<8x1x32xf16>
            %68 = loom.gather ins(%64 : tensor<1x32xf16>) outs(%67 : tensor<8x1x32xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x1x32xf16>
            loom.semaphore_give %35 : memref<1x32xf16>
            %69 = loom.alloc [8, 1, 32, 128] on @L1 : memref<8x1x32x128xf16>
            %70 = loom.semaphore_take %69 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
            %71 = loom.init_tensor %70[8, 1, 32, 128] : memref<8x1x32x128xf16> -> tensor<8x1x32x128xf16>
            %72 = loom.gather ins(%63#2 : tensor<1x32x128xf16>) outs(%71 : tensor<8x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x1x32x128xf16>
            loom.semaphore_give %29 : memref<1x32x128xf16>
            %73 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %73 {
              %74 = linalg.fill ins(%cst_2 : f16) outs(%34 : tensor<1x32xf16>) -> tensor<1x32xf16>
              %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%68 : tensor<8x1x32xf16>) outs(%74 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %82 = arith.maximumf %in, %out : f16
                linalg.yield %82 : f16
              } -> tensor<1x32xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%68, %75 : tensor<8x1x32xf16>, tensor<1x32xf16>) outs(%67 : tensor<8x1x32xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %82 = arith.subf %in, %in_9 : f16
                %83 = math.powf %cst, %82 : f16
                linalg.yield %83 : f16
              } -> tensor<8x1x32xf16>
              loom.semaphore_give %33 : memref<1x32xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%72, %76 : tensor<8x1x32x128xf16>, tensor<8x1x32xf16>) outs(%71 : tensor<8x1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %82 = arith.mulf %in, %in_9 : f16
                linalg.yield %82 : f16
              } -> tensor<8x1x32x128xf16>
              loom.semaphore_give %66 : memref<8x1x32xf16>
              %78 = linalg.fill ins(%cst_0 : f16) outs(%23 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%77 : tensor<8x1x32x128xf16>) outs(%78 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %82 = arith.addf %in, %out : f16
                linalg.yield %82 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %70 : memref<8x1x32x128xf16>
              %c0_6 = arith.constant 0 : index
              %c0_7 = arith.constant 0 : index
              %80 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_6, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%80], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %81 = loom.bufferize_to_memref %79 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              loom.copy %81, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.semaphore_give %22 : memref<1x32x128xf16>
            }
          } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
