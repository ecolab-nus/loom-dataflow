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
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_dim_x_level0_bc8__tile_b64__tile_n512__tile_s64(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c16 = arith.constant 16 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg6)
            %21 = arith.muli %arg4, %c64 : index
            %22 = loom.alloc [64, 32, 128] on @L1 : memref<64x32x128xf16>
            %23 = loom.semaphore_take %22 : memref<64x32x128xf16> -> memref<64x32x128xf16>
            %24 = loom.init_tensor %23[64, 32, 128] : memref<64x32x128xf16> -> tensor<64x32x128xf16>
            %25 = loom.semaphore_take %22 : memref<64x32x128xf16> -> memref<64x32x128xf16>
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%21, %c0_3, %c0_4)
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [64, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<64x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<64x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<64x32x128xf16>
            %27 = loom.bufferize_to_tensor %25[64, 32, 128] : memref<64x32x128xf16> -> tensor<64x32x128xf16>
            %28 = arith.muli %20, %c64 : index
            %29 = loom.alloc [64, 32, 128] on @L1 : memref<64x32x128xf16>
            %30 = loom.semaphore_take %29 : memref<64x32x128xf16> -> memref<64x32x128xf16>
            %31 = loom.init_tensor %30[64, 32, 128] : memref<64x32x128xf16> -> tensor<64x32x128xf16>
            %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<64x32x128xf16>) -> tensor<64x32x128xf16>
            %33 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
            %34 = loom.semaphore_take %33 : memref<64x32x1xf16> -> memref<64x32x1xf16>
            %35 = loom.init_tensor %34[64, 32, 1] : memref<64x32x1xf16> -> tensor<64x32x1xf16>
            %36 = loom.semaphore_take %33 : memref<64x32x1xf16> -> memref<64x32x1xf16>
            %37 = loom.init_tensor %36[64, 32, 1] : memref<64x32x1xf16> -> tensor<64x32x1xf16>
            %38 = loom.semaphore_take %33 : memref<64x32x1xf16> -> memref<64x32x1xf16>
            %39 = loom.init_tensor %38[64, 32, 1] : memref<64x32x1xf16> -> tensor<64x32x1xf16>
            %40 = loom.semaphore_take %33 : memref<64x32x1xf16> -> memref<64x32x1xf16>
            %41 = loom.init_tensor %40[64, 32, 1] : memref<64x32x1xf16> -> tensor<64x32x1xf16>
            %42 = linalg.fill ins(%cst_0 : f16) outs(%41 : tensor<64x32x1xf16>) -> tensor<64x32x1xf16>
            %43 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
            %44 = loom.semaphore_take %43 : memref<64x32x1xf16> -> memref<64x32x1xf16>
            %45 = loom.init_tensor %44[64, 32, 1] : memref<64x32x1xf16> -> tensor<64x32x1xf16>
            %46 = linalg.fill ins(%cst_1 : f16) outs(%45 : tensor<64x32x1xf16>) -> tensor<64x32x1xf16>
            %47 = loom.alloc [64, 32, 128] on @L1 : memref<64x32x128xf16>
            %48 = loom.semaphore_take %47 : memref<64x32x128xf16> -> memref<64x32x128xf16>
            %49 = loom.init_tensor %48[64, 32, 128] : memref<64x32x128xf16> -> tensor<64x32x128xf16>
            %50 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
            %51 = loom.semaphore_take %50 : memref<64x32x1xf16> -> memref<64x32x1xf16>
            %52 = loom.init_tensor %51[64, 32, 1] : memref<64x32x1xf16> -> tensor<64x32x1xf16>
            %53 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
            %54 = loom.semaphore_take %53 : memref<64x32x1xf16> -> memref<64x32x1xf16>
            %55 = loom.init_tensor %54[64, 32, 1] : memref<64x32x1xf16> -> tensor<64x32x1xf16>
            %56 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
            %57 = loom.semaphore_take %56 : memref<64x32x1xf16> -> memref<64x32x1xf16>
            %58 = loom.init_tensor %57[64, 32, 1] : memref<64x32x1xf16> -> tensor<64x32x1xf16>
            %59 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
            %60 = loom.semaphore_take %59 : memref<64x128x512xf16> -> memref<64x128x512xf16>
            %61 = loom.alloc [64, 32, 512] on @L1 : memref<64x32x512xf16>
            %62 = loom.semaphore_take %61 : memref<64x32x512xf16> -> memref<64x32x512xf16>
            %63 = loom.init_tensor %62[64, 32, 512] : memref<64x32x512xf16> -> tensor<64x32x512xf16>
            %64 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
            %65 = loom.semaphore_take %64 : memref<64x512x128xf16> -> memref<64x512x128xf16>
            %c0_5 = arith.constant 0 : index
            %66 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%21, %c0_5, %28)
            %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%66], sizes: [64, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<64x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_6, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<64x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<64x128x512xf16>
            %67 = loom.bufferize_to_tensor %60[64, 128, 512] : memref<64x128x512xf16> -> tensor<64x128x512xf16>
            %68 = linalg.fill ins(%cst : f16) outs(%63 : tensor<64x32x512xf16>) -> tensor<64x32x512xf16>
            %69 = linalg.batch_matmul ins(%27, %67 : tensor<64x32x128xf16>, tensor<64x128x512xf16>) outs(%68 : tensor<64x32x512xf16>) -> tensor<64x32x512xf16>
            loom.semaphore_give %60 : memref<64x128x512xf16>
            %70 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<64x32x1xf16>) -> tensor<64x32x1xf16>
            %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%69 : tensor<64x32x512xf16>) outs(%70 : tensor<64x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %100 = arith.maximumf %in, %out : f16
              linalg.yield %100 : f16
            } -> tensor<64x32x1xf16>
            %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %71 : tensor<64x32x1xf16>, tensor<64x32x1xf16>) outs(%52 : tensor<64x32x1xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %100 = arith.mulf %in_9, %cst_2 : f16
              %101 = arith.cmpf ogt, %in, %100 : f16
              %102 = arith.select %101, %in, %100 : f16
              linalg.yield %102 : f16
            } -> tensor<64x32x1xf16>
            %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69, %72 : tensor<64x32x512xf16>, tensor<64x32x1xf16>) outs(%63 : tensor<64x32x512xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %100 = arith.mulf %in, %cst_2 : f16
              %101 = arith.subf %100, %in_9 : f16
              %102 = math.exp %101 : f16
              linalg.yield %102 : f16
            } -> tensor<64x32x512xf16>
            %74 = linalg.fill ins(%cst : f16) outs(%55 : tensor<64x32x1xf16>) -> tensor<64x32x1xf16>
            %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%73 : tensor<64x32x512xf16>) outs(%74 : tensor<64x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %100 = arith.addf %in, %out : f16
              linalg.yield %100 : f16
            } -> tensor<64x32x1xf16>
            %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %72 : tensor<64x32x1xf16>, tensor<64x32x1xf16>) outs(%58 : tensor<64x32x1xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %100 = arith.subf %in, %in_9 : f16
              %101 = math.exp %100 : f16
              linalg.yield %101 : f16
            } -> tensor<64x32x1xf16>
            %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %76, %75 : tensor<64x32x1xf16>, tensor<64x32x1xf16>, tensor<64x32x1xf16>) outs(%42 : tensor<64x32x1xf16>) {
            ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
              %100 = arith.mulf %in, %in_9 : f16
              %101 = arith.addf %100, %in_10 : f16
              linalg.yield %101 : f16
            } -> tensor<64x32x1xf16>
            loom.semaphore_give %54 : memref<64x32x1xf16>
            %c0_7 = arith.constant 0 : index
            %78 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%21, %28, %c0_7)
            %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%78], sizes: [64, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<64x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_8, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<64x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<64x512x128xf16>
            %79 = loom.bufferize_to_tensor %65[64, 512, 128] : memref<64x512x128xf16> -> tensor<64x512x128xf16>
            %80 = linalg.fill ins(%cst : f16) outs(%49 : tensor<64x32x128xf16>) -> tensor<64x32x128xf16>
            %81 = linalg.batch_matmul ins(%73, %79 : tensor<64x32x512xf16>, tensor<64x512x128xf16>) outs(%80 : tensor<64x32x128xf16>) -> tensor<64x32x128xf16>
            loom.semaphore_give %65 : memref<64x512x128xf16>
            loom.semaphore_give %62 : memref<64x32x512xf16>
            %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %32, %76 : tensor<64x32x128xf16>, tensor<64x32x128xf16>, tensor<64x32x1xf16>) outs(%32 : tensor<64x32x128xf16>) {
            ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
              %100 = arith.mulf %in_9, %in_10 : f16
              %101 = arith.addf %in, %100 : f16
              linalg.yield %101 : f16
            } -> tensor<64x32x128xf16>
            loom.semaphore_give %57 : memref<64x32x1xf16>
            loom.semaphore_give %48 : memref<64x32x128xf16>
            %83 = linalg.copy ins(%72 : tensor<64x32x1xf16>) outs(%45 : tensor<64x32x1xf16>) -> tensor<64x32x1xf16>
            loom.semaphore_give %51 : memref<64x32x1xf16>
            loom.semaphore_give %25 : memref<64x32x128xf16>
            %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %83 : tensor<64x32x1xf16>, tensor<64x32x1xf16>) outs(%39 : tensor<64x32x1xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %100 = math.log %in : f16
              %101 = arith.addf %100, %in_9 : f16
              linalg.yield %101 : f16
            } -> tensor<64x32x1xf16>
            loom.semaphore_give %40 : memref<64x32x1xf16>
            loom.semaphore_give %44 : memref<64x32x1xf16>
            %85 = loom.alloc [128, 64, 32, 1] on @L1 : memref<128x64x32x1xf16>
            %86 = loom.semaphore_take %85 : memref<128x64x32x1xf16> -> memref<128x64x32x1xf16>
            %87 = loom.init_tensor %86[128, 64, 32, 1] : memref<128x64x32x1xf16> -> tensor<128x64x32x1xf16>
            %88 = loom.semaphore_take %33 : memref<64x32x1xf16> -> memref<64x32x1xf16>
            %89 = loom.init_tensor %88[64, 32, 1] : memref<64x32x1xf16> -> tensor<64x32x1xf16>
            %90 = loom.sync ins(%84 : tensor<64x32x1xf16>) outs(%89 : tensor<64x32x1xf16>) -> tensor<64x32x1xf16>
            %91 = loom.gather ins(%90 : tensor<64x32x1xf16>) outs(%87 : tensor<128x64x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<128x64x32x1xf16>
            loom.semaphore_give %88 : memref<64x32x1xf16>
            loom.semaphore_give %38 : memref<64x32x1xf16>
            %92 = loom.alloc [128, 64, 32, 128] on @L1 : memref<128x64x32x128xf16>
            %93 = loom.semaphore_take %92 : memref<128x64x32x128xf16> -> memref<128x64x32x128xf16>
            %94 = loom.init_tensor %93[128, 64, 32, 128] : memref<128x64x32x128xf16> -> tensor<128x64x32x128xf16>
            %95 = loom.semaphore_take %29 : memref<64x32x128xf16> -> memref<64x32x128xf16>
            %96 = loom.init_tensor %95[64, 32, 128] : memref<64x32x128xf16> -> tensor<64x32x128xf16>
            %97 = loom.sync ins(%82 : tensor<64x32x128xf16>) outs(%96 : tensor<64x32x128xf16>) -> tensor<64x32x128xf16>
            %98 = loom.gather ins(%97 : tensor<64x32x128xf16>) outs(%94 : tensor<128x64x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<128x64x32x128xf16>
            loom.semaphore_give %95 : memref<64x32x128xf16>
            loom.semaphore_give %30 : memref<64x32x128xf16>
            %99 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %99 {
              %100 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<64x32x1xf16>) -> tensor<64x32x1xf16>
              %101 = loom.semaphore_take %85 : memref<128x64x32x1xf16> -> memref<128x64x32x1xf16>
              %102 = loom.init_tensor %101[128, 64, 32, 1] : memref<128x64x32x1xf16> -> tensor<128x64x32x1xf16>
              %103 = loom.sync ins(%91 : tensor<128x64x32x1xf16>) outs(%102 : tensor<128x64x32x1xf16>) -> tensor<128x64x32x1xf16>
              %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<128x64x32x1xf16>) outs(%100 : tensor<64x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %119 = arith.maximumf %in, %out : f16
                linalg.yield %119 : f16
              } -> tensor<64x32x1xf16>
              %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %104 : tensor<128x64x32x1xf16>, tensor<64x32x1xf16>) outs(%87 : tensor<128x64x32x1xf16>) {
              ^bb0(%in: f16, %in_12: f16, %out: f16):
                %119 = arith.subf %in, %in_12 : f16
                %120 = math.exp %119 : f16
                linalg.yield %120 : f16
              } -> tensor<128x64x32x1xf16>
              loom.semaphore_give %36 : memref<64x32x1xf16>
              %106 = linalg.fill ins(%cst : f16) outs(%35 : tensor<64x32x1xf16>) -> tensor<64x32x1xf16>
              %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%105 : tensor<128x64x32x1xf16>) outs(%106 : tensor<64x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %119 = arith.addf %in, %out : f16
                linalg.yield %119 : f16
              } -> tensor<64x32x1xf16>
              %108 = loom.semaphore_take %92 : memref<128x64x32x128xf16> -> memref<128x64x32x128xf16>
              %109 = loom.init_tensor %108[128, 64, 32, 128] : memref<128x64x32x128xf16> -> tensor<128x64x32x128xf16>
              %110 = loom.sync ins(%98 : tensor<128x64x32x128xf16>) outs(%109 : tensor<128x64x32x128xf16>) -> tensor<128x64x32x128xf16>
              %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %105, %107 : tensor<128x64x32x128xf16>, tensor<128x64x32x1xf16>, tensor<64x32x1xf16>) outs(%94 : tensor<128x64x32x128xf16>) {
              ^bb0(%in: f16, %in_12: f16, %in_13: f16, %out: f16):
                %119 = arith.divf %in_12, %in_13 : f16
                %120 = arith.mulf %in, %119 : f16
                linalg.yield %120 : f16
              } -> tensor<128x64x32x128xf16>
              loom.semaphore_give %101 : memref<128x64x32x1xf16>
              loom.semaphore_give %86 : memref<128x64x32x1xf16>
              loom.semaphore_give %34 : memref<64x32x1xf16>
              %112 = linalg.fill ins(%cst : f16) outs(%24 : tensor<64x32x128xf16>) -> tensor<64x32x128xf16>
              %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<128x64x32x128xf16>) outs(%112 : tensor<64x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %119 = arith.addf %in, %out : f16
                linalg.yield %119 : f16
              } -> tensor<64x32x128xf16>
              loom.semaphore_give %108 : memref<128x64x32x128xf16>
              loom.semaphore_give %93 : memref<128x64x32x128xf16>
              %c0_9 = arith.constant 0 : index
              %c0_10 = arith.constant 0 : index
              %114 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%21, %c0_9, %c0_10)
              %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%114], sizes: [64, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<64x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %115 = loom.semaphore_take %22 : memref<64x32x128xf16> -> memref<64x32x128xf16>
              %116 = loom.init_tensor %115[64, 32, 128] : memref<64x32x128xf16> -> tensor<64x32x128xf16>
              %117 = loom.sync ins(%113 : tensor<64x32x128xf16>) outs(%116 : tensor<64x32x128xf16>) -> tensor<64x32x128xf16>
              %118 = loom.bufferize_to_memref %117 : tensor<64x32x128xf16> -> memref<64x32x128xf16>
              loom.copy %118, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<64x32x128xf16> to memref<64x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.semaphore_give %115 : memref<64x32x128xf16>
              loom.semaphore_give %23 : memref<64x32x128xf16>
            }
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
