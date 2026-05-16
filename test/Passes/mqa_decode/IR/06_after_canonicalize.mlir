module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.memory.bank "mem_bank", {bsize = 16 : i64, nblk = 5856 : i64}
  %4 = adl.spatial_dim "dim_nbank", 16
  %5 = adl.memory.array "mem_L1", [%4] of %3
  %6 = adl.resource.exclusive "res_matrix_lane"
  %7 = adl.resource.exclusive "res_vector_lane"
  %8 = adl.processor.compute @proc_matrix_lane, [(%5, %5)], with [%6]
  %9 = adl.processor.compute @proc_vector_lane, [(%5, %5)], with [%7]
  %10 = adl.arch.compose "arch_core", arch[%8, %9], mem[%5]
  %11 = adl.spatial_dim "dim_x", 8
  %12 = adl.spatial_dim "dim_y", 8
  %13 = adl.arch.scale "arch_mesh", [%11, %12] of %10
  %14 = adl.memory.array "mem_array_L1", [%11, %12] of %5
  %15 = adl.processor.dmover @proc_dram_l1_noc0, [(%2, %14)]
  %16 = adl.processor.dmover @proc_dram_l1_noc1, [(%14, %2), (%14, %14)]
  %17 = adl.arch.compose "arch_system", arch[%13, %15, %16], mem[%2]
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                %20 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %21 = loom.semaphore_take %20 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %c0_3 = arith.constant 0 : index
                %c0_4 = arith.constant 0 : index
                %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_3, %c0_4)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %23 = arith.addi %arg6, %arg4 : index
                loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %23], LR : [%c7, %23]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
                %24 = loom.bufferize_to_tensor %21[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %25 = arith.muli %19, %c512 : index
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
                %45 = loom.semaphore_take %42 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %46 = loom.init_tensor %45[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %47 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %48 = loom.semaphore_take %47 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %49 = loom.init_tensor %48[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %50 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %51 = loom.semaphore_take %50 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %52 = loom.init_tensor %51[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %53 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %54 = loom.semaphore_take %53 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %55 = loom.init_tensor %54[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %56 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %57 = loom.semaphore_take %56 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %58 = loom.init_tensor %57[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %59 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
                %60 = loom.semaphore_take %59 : memref<1x128x512xf16> -> memref<1x128x512xf16>
                %61 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %62 = loom.semaphore_take %61 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %63 = loom.init_tensor %62[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %64 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %65 = loom.semaphore_take %64 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %66 = loom.init_tensor %65[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %67 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
                %68 = loom.semaphore_take %67 : memref<1x512x128xf16> -> memref<1x512x128xf16>
                %c0_5 = arith.constant 0 : index
                %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %25)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%69], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %reinterpret_cast_6, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %23], LR : [%arg5, %23]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
                %70 = loom.bufferize_to_tensor %60[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %71 = linalg.fill ins(%cst : f16) outs(%63 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                %72 = linalg.batch_matmul ins(%24, %70 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%71 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                loom.semaphore_give %60 : memref<1x128x512xf16>
                %73 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%72 : tensor<1x32x512xf16>) outs(%73 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %116 = arith.maximumf %in, %out : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %74 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%52 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.mulf %in_9, %cst_2 : f16
                  %117 = arith.cmpf ogt, %in, %116 : f16
                  %118 = arith.select %117, %in, %116 : f16
                  linalg.yield %118 : f16
                } -> tensor<1x32x1xf16>
                %76 = loom.broadcast ins(%75 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72, %76 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%63 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.mulf %in, %cst_2 : f16
                  %117 = arith.subf %116, %in_9 : f16
                  %118 = math.exp %117 : f16
                  linalg.yield %118 : f16
                } -> tensor<1x32x512xf16>
                loom.semaphore_give %65 : memref<1x32x512xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%55 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.subf %in, %in_9 : f16
                  %117 = math.exp %116 : f16
                  linalg.yield %117 : f16
                } -> tensor<1x32x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%58 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.mulf %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<1x32x512xf16>) outs(%79 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %116 = arith.addf %in, %out : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %57 : memref<1x32x1xf16>
                %81 = loom.broadcast ins(%78 : tensor<1x32x1xf16>) outs(%46 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %54 : memref<1x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %81 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%46 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.mulf %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x128xf16>
                %c0_7 = arith.constant 0 : index
                %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %25, %c0_7)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%83], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_8, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %23], LR : [%arg5, %23]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %84 = loom.bufferize_to_tensor %68[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %85 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %86 = linalg.batch_matmul ins(%77, %84 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%85 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                loom.semaphore_give %68 : memref<1x512x128xf16>
                loom.semaphore_give %62 : memref<1x32x512xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %82 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%29 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.addf %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %48 : memref<1x32x128xf16>
                loom.semaphore_give %45 : memref<1x32x128xf16>
                %88 = linalg.copy ins(%75 : tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                loom.semaphore_give %51 : memref<1x32x1xf16>
                loom.semaphore_give %21 : memref<1x32x128xf16>
                %89 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %90 = loom.semaphore_take %89 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %91 = loom.init_tensor %90[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80, %88 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%91 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = math.log %in : f16
                  %117 = arith.addf %116, %in_9 : f16
                  linalg.yield %117 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %39 : memref<1x32x1xf16>
                %93 = loom.broadcast ins(%80 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %35 : memref<1x32x1xf16>
                %94 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %95 = loom.semaphore_take %94 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %96 = loom.init_tensor %95[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %93 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%96 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.divf %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %43 : memref<1x32x128xf16>
                loom.semaphore_give %27 : memref<1x32x128xf16>
                %98 = loom.bufferize_to_memref %92 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
                %99 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %100 = loom.semaphore_take %99 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                loom.gather %98, %100 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
                loom.semaphore_give %90 : memref<1x32x1xf16>
                %101 = loom.bufferize_to_tensor %100[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %102 = loom.bufferize_to_memref %97 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %103 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %104 = loom.semaphore_take %103 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                loom.gather %102, %104 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
                loom.semaphore_give %95 : memref<1x32x128xf16>
                %105 = loom.bufferize_to_tensor %104[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %106 = arith.cmpi eq, %19, %c0 : index
                %107 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %108 = loom.semaphore_take %107 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %109 = loom.init_tensor %108[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %110 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %111 = loom.semaphore_take %110 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %112 = loom.init_tensor %111[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %113 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %114 = loom.semaphore_take %113 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %115 = loom.init_tensor %114[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                scf.if %106 {
                  %116 = linalg.fill ins(%cst_1 : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%101 : tensor<16x1x32x1xf16>) outs(%116 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %128 = arith.maximumf %in, %out : f16
                    linalg.yield %128 : f16
                  } -> tensor<1x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %117 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%112 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %128 = arith.subf %in, %in_12 : f16
                    %129 = math.exp %128 : f16
                    linalg.yield %129 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %100 : memref<16x1x32x1xf16>
                  loom.semaphore_give %33 : memref<1x32x1xf16>
                  %119 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%118 : tensor<16x1x32x1xf16>) outs(%119 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %128 = arith.addf %in, %out : f16
                    linalg.yield %128 : f16
                  } -> tensor<1x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%118, %120 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%112 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %128 = arith.divf %in, %in_12 : f16
                    linalg.yield %128 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %31 : memref<1x32x1xf16>
                  %122 = loom.broadcast ins(%121 : tensor<16x1x32x1xf16>) outs(%115 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %111 : memref<16x1x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %122 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%115 : tensor<16x1x32x128xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %128 = arith.mulf %in, %in_12 : f16
                    linalg.yield %128 : f16
                  } -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %104 : memref<16x1x32x128xf16>
                  %124 = linalg.fill ins(%cst : f16) outs(%109 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%123 : tensor<16x1x32x128xf16>) outs(%124 : tensor<1x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %128 = arith.addf %in, %out : f16
                    linalg.yield %128 : f16
                  } -> tensor<1x32x128xf16>
                  loom.semaphore_give %114 : memref<16x1x32x128xf16>
                  %c0_9 = arith.constant 0 : index
                  %c0_10 = arith.constant 0 : index
                  %126 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%126], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %127 = loom.bufferize_to_memref %125 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                  loom.copy %127, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %23], LR : [%arg5, %23]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %108 : memref<1x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %20 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %21 = loom.semaphore_take %20 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %23 = arith.muli %arg4, %c2 : index
              %24 = arith.addi %23, %c1 : index
              loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %25 = loom.bufferize_to_tensor %21[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %26 = arith.muli %19, %c512 : index
              %27 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %28 = loom.semaphore_take %27 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
              %46 = loom.semaphore_take %43 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %47 = loom.init_tensor %46[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %48 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %50 = loom.init_tensor %49[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %56 = loom.init_tensor %55[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %57 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %58 = loom.semaphore_take %57 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %59 = loom.init_tensor %58[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %60 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %61 = loom.semaphore_take %60 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %62 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %64 = loom.init_tensor %63[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %65 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %66 = loom.semaphore_take %65 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %67 = loom.init_tensor %66[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %68 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %69 = loom.semaphore_take %68 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %70 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %26)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%70], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %71 = arith.addi %arg6, %23 : index
              loom.copy %reinterpret_cast_6, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %71], LR : [%arg5, %71]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %72 = loom.bufferize_to_tensor %61[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %73 = linalg.fill ins(%cst : f16) outs(%64 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %74 = linalg.batch_matmul ins(%25, %72 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %61 : memref<1x128x512xf16>
              %75 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x32x512xf16>) outs(%75 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.maximumf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in_9, %cst_2 : f16
                %119 = arith.cmpf ogt, %in, %118 : f16
                %120 = arith.select %119, %in, %118 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x1xf16>
              %78 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%64 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %cst_2 : f16
                %119 = arith.subf %118, %in_9 : f16
                %120 = math.exp %119 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %66 : memref<1x32x512xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.subf %in, %in_9 : f16
                %119 = math.exp %118 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %80 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%59 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x32x512xf16>) outs(%81 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.addf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %58 : memref<1x32x1xf16>
              %83 = loom.broadcast ins(%80 : tensor<1x32x1xf16>) outs(%47 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%47 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %85 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %26, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%85], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %71], LR : [%arg5, %71]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %86 = loom.bufferize_to_tensor %69[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %87 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %88 = linalg.batch_matmul ins(%79, %86 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%87 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %69 : memref<1x512x128xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %84 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.addf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %49 : memref<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %90 = linalg.copy ins(%77 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              loom.semaphore_give %21 : memref<1x32x128xf16>
              %91 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %92 = loom.semaphore_take %91 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %93 = loom.init_tensor %92[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %90 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%93 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = math.log %in : f16
                %119 = arith.addf %118, %in_9 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %95 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%45 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %96 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %97 = loom.semaphore_take %96 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %98 = loom.init_tensor %97[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %95 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%98 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.divf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %100 = loom.bufferize_to_memref %94 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %101 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %102 = loom.semaphore_take %101 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              loom.gather %100, %102 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 2] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %92 : memref<1x32x1xf16>
              %103 = loom.bufferize_to_tensor %102[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = loom.bufferize_to_memref %99 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %104, %106 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 2] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %97 : memref<1x32x128xf16>
              %107 = loom.bufferize_to_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = arith.cmpi eq, %19, %c0 : index
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %108 {
                %118 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<16x1x32x1xf16>) outs(%118 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.maximumf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %119 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.subf %in, %in_12 : f16
                  %132 = math.exp %131 : f16
                  linalg.yield %132 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %102 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %121 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%120 : tensor<16x1x32x1xf16>) outs(%121 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %122 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.divf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %32 : memref<1x32x1xf16>
                %124 = loom.broadcast ins(%123 : tensor<16x1x32x1xf16>) outs(%117 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %124 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.mulf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x128xf16>) outs(%126 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %128 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%128], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %129 = loom.bufferize_to_memref %127 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %130 = arith.addi %arg6, %23 : index
                loom.copy %129, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %130], LR : [%arg5, %130]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %110 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %20 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %21 = loom.semaphore_take %20 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %23 = arith.muli %arg4, %c4 : index
              %24 = arith.addi %23, %c3 : index
              loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %25 = loom.bufferize_to_tensor %21[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %26 = arith.muli %19, %c512 : index
              %27 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %28 = loom.semaphore_take %27 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
              %46 = loom.semaphore_take %43 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %47 = loom.init_tensor %46[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %48 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %50 = loom.init_tensor %49[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %56 = loom.init_tensor %55[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %57 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %58 = loom.semaphore_take %57 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %59 = loom.init_tensor %58[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %60 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %61 = loom.semaphore_take %60 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %62 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %64 = loom.init_tensor %63[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %65 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %66 = loom.semaphore_take %65 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %67 = loom.init_tensor %66[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %68 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %69 = loom.semaphore_take %68 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %70 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %26)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%70], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %71 = arith.addi %arg6, %23 : index
              loom.copy %reinterpret_cast_6, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %71], LR : [%arg5, %71]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %72 = loom.bufferize_to_tensor %61[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %73 = linalg.fill ins(%cst : f16) outs(%64 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %74 = linalg.batch_matmul ins(%25, %72 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %61 : memref<1x128x512xf16>
              %75 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x32x512xf16>) outs(%75 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.maximumf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in_9, %cst_2 : f16
                %119 = arith.cmpf ogt, %in, %118 : f16
                %120 = arith.select %119, %in, %118 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x1xf16>
              %78 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%64 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %cst_2 : f16
                %119 = arith.subf %118, %in_9 : f16
                %120 = math.exp %119 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %66 : memref<1x32x512xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.subf %in, %in_9 : f16
                %119 = math.exp %118 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %80 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%59 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x32x512xf16>) outs(%81 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.addf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %58 : memref<1x32x1xf16>
              %83 = loom.broadcast ins(%80 : tensor<1x32x1xf16>) outs(%47 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%47 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %85 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %26, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%85], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %71], LR : [%arg5, %71]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %86 = loom.bufferize_to_tensor %69[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %87 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %88 = linalg.batch_matmul ins(%79, %86 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%87 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %69 : memref<1x512x128xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %84 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.addf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %49 : memref<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %90 = linalg.copy ins(%77 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              loom.semaphore_give %21 : memref<1x32x128xf16>
              %91 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %92 = loom.semaphore_take %91 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %93 = loom.init_tensor %92[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %90 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%93 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = math.log %in : f16
                %119 = arith.addf %118, %in_9 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %95 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%45 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %96 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %97 = loom.semaphore_take %96 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %98 = loom.init_tensor %97[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %95 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%98 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.divf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %100 = loom.bufferize_to_memref %94 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %101 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %102 = loom.semaphore_take %101 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              loom.gather %100, %102 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 4] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %92 : memref<1x32x1xf16>
              %103 = loom.bufferize_to_tensor %102[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = loom.bufferize_to_memref %99 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %104, %106 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 4] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %97 : memref<1x32x128xf16>
              %107 = loom.bufferize_to_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = arith.cmpi eq, %19, %c0 : index
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %108 {
                %118 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<16x1x32x1xf16>) outs(%118 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.maximumf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %119 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.subf %in, %in_12 : f16
                  %132 = math.exp %131 : f16
                  linalg.yield %132 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %102 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %121 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%120 : tensor<16x1x32x1xf16>) outs(%121 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %122 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.divf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %32 : memref<1x32x1xf16>
                %124 = loom.broadcast ins(%123 : tensor<16x1x32x1xf16>) outs(%117 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %124 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.mulf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x128xf16>) outs(%126 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %128 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%128], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %129 = loom.bufferize_to_memref %127 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %130 = arith.addi %arg6, %23 : index
                loom.copy %129, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %130], LR : [%arg5, %130]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %110 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %19 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %20 = loom.semaphore_take %19 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast, %20 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %22 = loom.bufferize_to_tensor %20[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %23 = arith.muli %18, %c512 : index
              %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %26 = loom.init_tensor %25[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %27 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %28 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %29 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %30 = loom.init_tensor %29[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %31 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %32 = loom.init_tensor %31[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %33 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %34 = loom.init_tensor %33[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %35 = linalg.fill ins(%cst_0 : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %36 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %37 = loom.semaphore_take %36 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %39 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %40 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %41 = loom.semaphore_take %40 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %42 = loom.init_tensor %41[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %43 = loom.semaphore_take %40 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %44 = loom.init_tensor %43[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
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
              %62 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %64 = loom.init_tensor %63[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %65 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %66 = loom.semaphore_take %65 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%arg7, %c0_5, %23)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%67], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %68 = arith.muli %arg4, %c8 : index
              %69 = arith.addi %arg6, %68 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %69], LR : [%arg5, %69]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %70 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %71 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %72 = linalg.batch_matmul ins(%22, %70 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%71 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              %73 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%72 : tensor<1x32x512xf16>) outs(%73 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.maximumf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %74 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in_9, %cst_2 : f16
                %119 = arith.cmpf ogt, %in, %118 : f16
                %120 = arith.select %119, %in, %118 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x1xf16>
              %76 = loom.broadcast ins(%75 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72, %76 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %cst_2 : f16
                %119 = arith.subf %118, %in_9 : f16
                %120 = math.exp %119 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.subf %in, %in_9 : f16
                %119 = math.exp %118 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<1x32x512xf16>) outs(%79 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.addf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %81 = loom.broadcast ins(%78 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %81 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%44 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%arg7, %23, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%83], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %69], LR : [%arg5, %69]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %84 = loom.bufferize_to_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %85 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %86 = linalg.batch_matmul ins(%77, %84 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%85 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %82 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%27 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.addf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              loom.semaphore_give %43 : memref<1x32x128xf16>
              %88 = linalg.copy ins(%75 : tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              loom.semaphore_give %20 : memref<1x32x128xf16>
              %89 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %90 = loom.semaphore_take %89 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %91 = loom.init_tensor %90[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80, %88 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%91 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = math.log %in : f16
                %119 = arith.addf %118, %in_9 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %37 : memref<1x32x1xf16>
              %93 = loom.broadcast ins(%80 : tensor<1x32x1xf16>) outs(%42 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %33 : memref<1x32x1xf16>
              %94 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %95 = loom.semaphore_take %94 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %96 = loom.init_tensor %95[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %93 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%96 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.divf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %41 : memref<1x32x128xf16>
              loom.semaphore_give %25 : memref<1x32x128xf16>
              %98 = loom.bufferize_to_memref %92 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %99 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %100 = loom.semaphore_take %99 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %101 = arith.muli %arg4, %c8 : index
              %102 = arith.addi %101, %c7 : index
              loom.gather %98, %100 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%18 : index), area : [8, 8] region : (UL : [%c0, %101], LR : [%c7, %102]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %90 : memref<1x32x1xf16>
              %103 = loom.bufferize_to_tensor %100[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = loom.bufferize_to_memref %97 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %104, %106 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%18 : index), area : [8, 8] region : (UL : [%c0, %101], LR : [%c7, %102]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %95 : memref<1x32x128xf16>
              %107 = loom.bufferize_to_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = arith.cmpi eq, %18, %c0 : index
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %108 {
                %118 = linalg.fill ins(%cst_1 : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<16x1x32x1xf16>) outs(%118 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.maximumf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %119 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.subf %in, %in_12 : f16
                  %132 = math.exp %131 : f16
                  linalg.yield %132 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %100 : memref<16x1x32x1xf16>
                loom.semaphore_give %31 : memref<1x32x1xf16>
                %121 = linalg.fill ins(%cst : f16) outs(%30 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%120 : tensor<16x1x32x1xf16>) outs(%121 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %122 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.divf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %29 : memref<1x32x1xf16>
                %124 = loom.broadcast ins(%123 : tensor<16x1x32x1xf16>) outs(%117 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %124 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.mulf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x128xf16>) outs(%126 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %128 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%128], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %129 = loom.bufferize_to_memref %127 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %130 = arith.addi %arg6, %101 : index
                loom.copy %129, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %130], LR : [%arg5, %130]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %110 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg6, %arg8)
                %20 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %21 = loom.semaphore_take %20 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %c0_3 = arith.constant 0 : index
                %c0_4 = arith.constant 0 : index
                %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_3, %c0_4)
                %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %23 = arith.addi %arg5, %arg4 : index
                loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%23, %c0], LR : [%23, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
                %24 = loom.bufferize_to_tensor %21[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %25 = arith.muli %19, %c512 : index
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
                %45 = loom.semaphore_take %42 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %46 = loom.init_tensor %45[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %47 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %48 = loom.semaphore_take %47 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %49 = loom.init_tensor %48[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %50 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %51 = loom.semaphore_take %50 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %52 = loom.init_tensor %51[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %53 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %54 = loom.semaphore_take %53 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %55 = loom.init_tensor %54[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %56 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %57 = loom.semaphore_take %56 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %58 = loom.init_tensor %57[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %59 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
                %60 = loom.semaphore_take %59 : memref<1x128x512xf16> -> memref<1x128x512xf16>
                %61 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %62 = loom.semaphore_take %61 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %63 = loom.init_tensor %62[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %64 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %65 = loom.semaphore_take %64 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %66 = loom.init_tensor %65[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %67 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
                %68 = loom.semaphore_take %67 : memref<1x512x128xf16> -> memref<1x512x128xf16>
                %c0_5 = arith.constant 0 : index
                %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %25)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%69], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %reinterpret_cast_6, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%23, %arg6], LR : [%23, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
                %70 = loom.bufferize_to_tensor %60[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %71 = linalg.fill ins(%cst : f16) outs(%63 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                %72 = linalg.batch_matmul ins(%24, %70 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%71 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                loom.semaphore_give %60 : memref<1x128x512xf16>
                %73 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%72 : tensor<1x32x512xf16>) outs(%73 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %116 = arith.maximumf %in, %out : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %74 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%52 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.mulf %in_9, %cst_2 : f16
                  %117 = arith.cmpf ogt, %in, %116 : f16
                  %118 = arith.select %117, %in, %116 : f16
                  linalg.yield %118 : f16
                } -> tensor<1x32x1xf16>
                %76 = loom.broadcast ins(%75 : tensor<1x32x1xf16>) outs(%66 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72, %76 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%63 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.mulf %in, %cst_2 : f16
                  %117 = arith.subf %116, %in_9 : f16
                  %118 = math.exp %117 : f16
                  linalg.yield %118 : f16
                } -> tensor<1x32x512xf16>
                loom.semaphore_give %65 : memref<1x32x512xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%55 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.subf %in, %in_9 : f16
                  %117 = math.exp %116 : f16
                  linalg.yield %117 : f16
                } -> tensor<1x32x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%58 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.mulf %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<1x32x512xf16>) outs(%79 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %116 = arith.addf %in, %out : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %57 : memref<1x32x1xf16>
                %81 = loom.broadcast ins(%78 : tensor<1x32x1xf16>) outs(%46 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %54 : memref<1x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %81 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%46 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.mulf %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x128xf16>
                %c0_7 = arith.constant 0 : index
                %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %25, %c0_7)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%83], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_8, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%23, %arg6], LR : [%23, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %84 = loom.bufferize_to_tensor %68[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %85 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %86 = linalg.batch_matmul ins(%77, %84 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%85 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                loom.semaphore_give %68 : memref<1x512x128xf16>
                loom.semaphore_give %62 : memref<1x32x512xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %82 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%29 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.addf %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %48 : memref<1x32x128xf16>
                loom.semaphore_give %45 : memref<1x32x128xf16>
                %88 = linalg.copy ins(%75 : tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                loom.semaphore_give %51 : memref<1x32x1xf16>
                loom.semaphore_give %21 : memref<1x32x128xf16>
                %89 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %90 = loom.semaphore_take %89 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %91 = loom.init_tensor %90[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80, %88 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%91 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = math.log %in : f16
                  %117 = arith.addf %116, %in_9 : f16
                  linalg.yield %117 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %39 : memref<1x32x1xf16>
                %93 = loom.broadcast ins(%80 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %35 : memref<1x32x1xf16>
                %94 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %95 = loom.semaphore_take %94 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %96 = loom.init_tensor %95[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %93 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%96 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %116 = arith.divf %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %43 : memref<1x32x128xf16>
                loom.semaphore_give %27 : memref<1x32x128xf16>
                %98 = loom.bufferize_to_memref %92 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
                %99 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %100 = loom.semaphore_take %99 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                loom.gather %98, %100 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
                loom.semaphore_give %90 : memref<1x32x1xf16>
                %101 = loom.bufferize_to_tensor %100[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %102 = loom.bufferize_to_memref %97 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %103 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %104 = loom.semaphore_take %103 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                loom.gather %102, %104 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
                loom.semaphore_give %95 : memref<1x32x128xf16>
                %105 = loom.bufferize_to_tensor %104[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %106 = arith.cmpi eq, %19, %c0 : index
                %107 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %108 = loom.semaphore_take %107 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %109 = loom.init_tensor %108[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %110 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %111 = loom.semaphore_take %110 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %112 = loom.init_tensor %111[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %113 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %114 = loom.semaphore_take %113 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %115 = loom.init_tensor %114[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                scf.if %106 {
                  %116 = linalg.fill ins(%cst_1 : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%101 : tensor<16x1x32x1xf16>) outs(%116 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %128 = arith.maximumf %in, %out : f16
                    linalg.yield %128 : f16
                  } -> tensor<1x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %117 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%112 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %128 = arith.subf %in, %in_12 : f16
                    %129 = math.exp %128 : f16
                    linalg.yield %129 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %100 : memref<16x1x32x1xf16>
                  loom.semaphore_give %33 : memref<1x32x1xf16>
                  %119 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%118 : tensor<16x1x32x1xf16>) outs(%119 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %128 = arith.addf %in, %out : f16
                    linalg.yield %128 : f16
                  } -> tensor<1x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%118, %120 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%112 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %128 = arith.divf %in, %in_12 : f16
                    linalg.yield %128 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %31 : memref<1x32x1xf16>
                  %122 = loom.broadcast ins(%121 : tensor<16x1x32x1xf16>) outs(%115 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %111 : memref<16x1x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %122 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%115 : tensor<16x1x32x128xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %128 = arith.mulf %in, %in_12 : f16
                    linalg.yield %128 : f16
                  } -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %104 : memref<16x1x32x128xf16>
                  %124 = linalg.fill ins(%cst : f16) outs(%109 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%123 : tensor<16x1x32x128xf16>) outs(%124 : tensor<1x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %128 = arith.addf %in, %out : f16
                    linalg.yield %128 : f16
                  } -> tensor<1x32x128xf16>
                  loom.semaphore_give %114 : memref<16x1x32x128xf16>
                  %c0_9 = arith.constant 0 : index
                  %c0_10 = arith.constant 0 : index
                  %126 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%126], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %127 = loom.bufferize_to_memref %125 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                  loom.copy %127, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%23, %arg6], LR : [%23, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg5, %arg6)
              %20 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %21 = loom.semaphore_take %20 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %23 = arith.muli %arg4, %c2 : index
              %24 = arith.addi %23, %c1 : index
              loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %25 = loom.bufferize_to_tensor %21[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %26 = arith.muli %19, %c512 : index
              %27 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %28 = loom.semaphore_take %27 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
              %46 = loom.semaphore_take %43 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %47 = loom.init_tensor %46[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %48 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %50 = loom.init_tensor %49[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %56 = loom.init_tensor %55[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %57 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %58 = loom.semaphore_take %57 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %59 = loom.init_tensor %58[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %60 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %61 = loom.semaphore_take %60 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %62 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %64 = loom.init_tensor %63[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %65 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %66 = loom.semaphore_take %65 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %67 = loom.init_tensor %66[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %68 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %69 = loom.semaphore_take %68 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %70 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %26)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%70], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %71 = arith.addi %arg5, %23 : index
              loom.copy %reinterpret_cast_6, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%71, %arg6], LR : [%71, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %72 = loom.bufferize_to_tensor %61[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %73 = linalg.fill ins(%cst : f16) outs(%64 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %74 = linalg.batch_matmul ins(%25, %72 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %61 : memref<1x128x512xf16>
              %75 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x32x512xf16>) outs(%75 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.maximumf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in_9, %cst_2 : f16
                %119 = arith.cmpf ogt, %in, %118 : f16
                %120 = arith.select %119, %in, %118 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x1xf16>
              %78 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%64 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %cst_2 : f16
                %119 = arith.subf %118, %in_9 : f16
                %120 = math.exp %119 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %66 : memref<1x32x512xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.subf %in, %in_9 : f16
                %119 = math.exp %118 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %80 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%59 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x32x512xf16>) outs(%81 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.addf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %58 : memref<1x32x1xf16>
              %83 = loom.broadcast ins(%80 : tensor<1x32x1xf16>) outs(%47 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%47 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %85 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %26, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%85], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%71, %arg6], LR : [%71, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %86 = loom.bufferize_to_tensor %69[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %87 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %88 = linalg.batch_matmul ins(%79, %86 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%87 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %69 : memref<1x512x128xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %84 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.addf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %49 : memref<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %90 = linalg.copy ins(%77 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              loom.semaphore_give %21 : memref<1x32x128xf16>
              %91 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %92 = loom.semaphore_take %91 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %93 = loom.init_tensor %92[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %90 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%93 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = math.log %in : f16
                %119 = arith.addf %118, %in_9 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %95 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%45 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %96 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %97 = loom.semaphore_take %96 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %98 = loom.init_tensor %97[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %95 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%98 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.divf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %100 = loom.bufferize_to_memref %94 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %101 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %102 = loom.semaphore_take %101 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              loom.gather %100, %102 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [2, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %92 : memref<1x32x1xf16>
              %103 = loom.bufferize_to_tensor %102[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = loom.bufferize_to_memref %99 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %104, %106 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [2, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %97 : memref<1x32x128xf16>
              %107 = loom.bufferize_to_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = arith.cmpi eq, %19, %c0 : index
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %108 {
                %118 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<16x1x32x1xf16>) outs(%118 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.maximumf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %119 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.subf %in, %in_12 : f16
                  %132 = math.exp %131 : f16
                  linalg.yield %132 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %102 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %121 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%120 : tensor<16x1x32x1xf16>) outs(%121 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %122 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.divf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %32 : memref<1x32x1xf16>
                %124 = loom.broadcast ins(%123 : tensor<16x1x32x1xf16>) outs(%117 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %124 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.mulf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x128xf16>) outs(%126 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %128 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%128], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %129 = loom.bufferize_to_memref %127 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %130 = arith.addi %arg5, %23 : index
                loom.copy %129, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%130, %arg6], LR : [%130, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %110 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%arg5, %arg6)
              %20 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %21 = loom.semaphore_take %20 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %23 = arith.muli %arg4, %c4 : index
              %24 = arith.addi %23, %c3 : index
              loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %25 = loom.bufferize_to_tensor %21[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %26 = arith.muli %19, %c512 : index
              %27 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %28 = loom.semaphore_take %27 : memref<1x32x128xf16> -> memref<1x32x128xf16>
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
              %46 = loom.semaphore_take %43 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %47 = loom.init_tensor %46[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %48 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %49 = loom.semaphore_take %48 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %50 = loom.init_tensor %49[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %54 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %55 = loom.semaphore_take %54 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %56 = loom.init_tensor %55[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %57 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %58 = loom.semaphore_take %57 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %59 = loom.init_tensor %58[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %60 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %61 = loom.semaphore_take %60 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %62 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %64 = loom.init_tensor %63[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %65 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %66 = loom.semaphore_take %65 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %67 = loom.init_tensor %66[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %68 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %69 = loom.semaphore_take %68 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %70 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %26)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%70], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %71 = arith.addi %arg5, %23 : index
              loom.copy %reinterpret_cast_6, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%71, %arg6], LR : [%71, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %72 = loom.bufferize_to_tensor %61[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %73 = linalg.fill ins(%cst : f16) outs(%64 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %74 = linalg.batch_matmul ins(%25, %72 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %61 : memref<1x128x512xf16>
              %75 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x32x512xf16>) outs(%75 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.maximumf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %76 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in_9, %cst_2 : f16
                %119 = arith.cmpf ogt, %in, %118 : f16
                %120 = arith.select %119, %in, %118 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x1xf16>
              %78 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%64 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %cst_2 : f16
                %119 = arith.subf %118, %in_9 : f16
                %120 = math.exp %119 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %66 : memref<1x32x512xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.subf %in, %in_9 : f16
                %119 = math.exp %118 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %80 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%59 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x32x512xf16>) outs(%81 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.addf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %58 : memref<1x32x1xf16>
              %83 = loom.broadcast ins(%80 : tensor<1x32x1xf16>) outs(%47 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%47 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %85 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %26, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%85], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%71, %arg6], LR : [%71, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %86 = loom.bufferize_to_tensor %69[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %87 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %88 = linalg.batch_matmul ins(%79, %86 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%87 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %69 : memref<1x512x128xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %84 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.addf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %49 : memref<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %90 = linalg.copy ins(%77 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              loom.semaphore_give %21 : memref<1x32x128xf16>
              %91 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %92 = loom.semaphore_take %91 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %93 = loom.init_tensor %92[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %90 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%93 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = math.log %in : f16
                %119 = arith.addf %118, %in_9 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %95 = loom.broadcast ins(%82 : tensor<1x32x1xf16>) outs(%45 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %96 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %97 = loom.semaphore_take %96 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %98 = loom.init_tensor %97[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %95 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%98 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.divf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %100 = loom.bufferize_to_memref %94 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %101 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %102 = loom.semaphore_take %101 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              loom.gather %100, %102 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [4, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %92 : memref<1x32x1xf16>
              %103 = loom.bufferize_to_tensor %102[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = loom.bufferize_to_memref %99 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %104, %106 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [4, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %97 : memref<1x32x128xf16>
              %107 = loom.bufferize_to_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = arith.cmpi eq, %19, %c0 : index
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %108 {
                %118 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<16x1x32x1xf16>) outs(%118 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.maximumf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %119 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.subf %in, %in_12 : f16
                  %132 = math.exp %131 : f16
                  linalg.yield %132 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %102 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %121 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%120 : tensor<16x1x32x1xf16>) outs(%121 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %122 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.divf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %32 : memref<1x32x1xf16>
                %124 = loom.broadcast ins(%123 : tensor<16x1x32x1xf16>) outs(%117 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %124 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.mulf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x128xf16>) outs(%126 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %128 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%128], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %129 = loom.bufferize_to_memref %127 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %130 = arith.addi %arg5, %23 : index
                loom.copy %129, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%130, %arg6], LR : [%130, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %110 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %19 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %20 = loom.semaphore_take %19 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %c0_3 = arith.constant 0 : index
              %c0_4 = arith.constant 0 : index
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_3, %c0_4)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%21], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast, %20 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
              %22 = loom.bufferize_to_tensor %20[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %23 = arith.muli %18, %c512 : index
              %24 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %26 = loom.init_tensor %25[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %27 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %28 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %29 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %30 = loom.init_tensor %29[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %31 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %32 = loom.init_tensor %31[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %33 = loom.semaphore_take %28 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %34 = loom.init_tensor %33[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %35 = linalg.fill ins(%cst_0 : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %36 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %37 = loom.semaphore_take %36 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %39 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %40 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %41 = loom.semaphore_take %40 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %42 = loom.init_tensor %41[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %43 = loom.semaphore_take %40 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %44 = loom.init_tensor %43[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
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
              %62 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %63 = loom.semaphore_take %62 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %64 = loom.init_tensor %63[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %65 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %66 = loom.semaphore_take %65 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%arg7, %c0_5, %23)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%67], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %68 = arith.muli %arg4, %c8 : index
              %69 = arith.addi %arg5, %68 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%69, %arg6], LR : [%69, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %70 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %71 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %72 = linalg.batch_matmul ins(%22, %70 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%71 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              %73 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%72 : tensor<1x32x512xf16>) outs(%73 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.maximumf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %74 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in_9, %cst_2 : f16
                %119 = arith.cmpf ogt, %in, %118 : f16
                %120 = arith.select %119, %in, %118 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x1xf16>
              %76 = loom.broadcast ins(%75 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72, %76 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %cst_2 : f16
                %119 = arith.subf %118, %in_9 : f16
                %120 = math.exp %119 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.subf %in, %in_9 : f16
                %119 = math.exp %118 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %78 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<1x32x512xf16>) outs(%79 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.addf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %81 = loom.broadcast ins(%78 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %81 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%44 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.mulf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%arg7, %23, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%83], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%69, %arg6], LR : [%69, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %84 = loom.bufferize_to_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %85 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %86 = linalg.batch_matmul ins(%77, %84 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%85 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %82 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%27 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.addf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              loom.semaphore_give %43 : memref<1x32x128xf16>
              %88 = linalg.copy ins(%75 : tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              loom.semaphore_give %20 : memref<1x32x128xf16>
              %89 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %90 = loom.semaphore_take %89 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %91 = loom.init_tensor %90[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80, %88 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%91 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = math.log %in : f16
                %119 = arith.addf %118, %in_9 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %37 : memref<1x32x1xf16>
              %93 = loom.broadcast ins(%80 : tensor<1x32x1xf16>) outs(%42 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %33 : memref<1x32x1xf16>
              %94 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %95 = loom.semaphore_take %94 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %96 = loom.init_tensor %95[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %93 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%96 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %118 = arith.divf %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %41 : memref<1x32x128xf16>
              loom.semaphore_give %25 : memref<1x32x128xf16>
              %98 = loom.bufferize_to_memref %92 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %99 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %100 = loom.semaphore_take %99 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %101 = arith.muli %arg4, %c8 : index
              %102 = arith.addi %101, %c7 : index
              loom.gather %98, %100 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%18 : index), area : [8, 8] region : (UL : [%101, %c0], LR : [%102, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %90 : memref<1x32x1xf16>
              %103 = loom.bufferize_to_tensor %100[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %104 = loom.bufferize_to_memref %97 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %105 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %106 = loom.semaphore_take %105 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %104, %106 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%18 : index), area : [8, 8] region : (UL : [%101, %c0], LR : [%102, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %95 : memref<1x32x128xf16>
              %107 = loom.bufferize_to_tensor %106[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %108 = arith.cmpi eq, %18, %c0 : index
              %109 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %110 = loom.semaphore_take %109 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %111 = loom.init_tensor %110[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %112 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %113 = loom.semaphore_take %112 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %114 = loom.init_tensor %113[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %115 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %116 = loom.semaphore_take %115 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %117 = loom.init_tensor %116[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %108 {
                %118 = linalg.fill ins(%cst_1 : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<16x1x32x1xf16>) outs(%118 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.maximumf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%103, %119 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.subf %in, %in_12 : f16
                  %132 = math.exp %131 : f16
                  linalg.yield %132 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %100 : memref<16x1x32x1xf16>
                loom.semaphore_give %31 : memref<1x32x1xf16>
                %121 = linalg.fill ins(%cst : f16) outs(%30 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%120 : tensor<16x1x32x1xf16>) outs(%121 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x1xf16>
                %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%120, %122 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%114 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.divf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %29 : memref<1x32x1xf16>
                %124 = loom.broadcast ins(%123 : tensor<16x1x32x1xf16>) outs(%117 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %113 : memref<16x1x32x1xf16>
                %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %124 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%117 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %131 = arith.mulf %in, %in_12 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %106 : memref<16x1x32x128xf16>
                %126 = linalg.fill ins(%cst : f16) outs(%111 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %127 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%125 : tensor<16x1x32x128xf16>) outs(%126 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %131 = arith.addf %in, %out : f16
                  linalg.yield %131 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %116 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %128 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%128], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %129 = loom.bufferize_to_memref %127 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %130 = arith.addi %arg5, %101 : index
                loom.copy %129, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%130, %arg6], LR : [%130, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
