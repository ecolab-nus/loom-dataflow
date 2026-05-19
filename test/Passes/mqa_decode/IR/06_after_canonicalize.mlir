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
                %56 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
                %57 = loom.semaphore_take %56 : memref<1x128x512xf16> -> memref<1x128x512xf16>
                %58 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %59 = loom.semaphore_take %58 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %60 = loom.init_tensor %59[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %61 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %62 = loom.semaphore_take %61 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %63 = loom.init_tensor %62[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %64 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
                %65 = loom.semaphore_take %64 : memref<1x512x128xf16> -> memref<1x512x128xf16>
                %c0_5 = arith.constant 0 : index
                %66 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %25)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%66], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %reinterpret_cast_6, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %23], LR : [%arg5, %23]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
                %67 = loom.bufferize_to_tensor %57[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %68 = linalg.fill ins(%cst : f16) outs(%60 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                %69 = linalg.batch_matmul ins(%24, %67 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%68 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                loom.semaphore_give %57 : memref<1x128x512xf16>
                %70 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%69 : tensor<1x32x512xf16>) outs(%70 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %115 = arith.maximumf %in, %out : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x1xf16>
                %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x32x1xf16>) outs(%52 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %115 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x1xf16>
                %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %72 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%52 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.cmpf ogt, %in, %in_9 : f16
                  %116 = arith.select %115, %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69 : tensor<1x32x512xf16>) outs(%60 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %115 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x512xf16>
                %75 = loom.broadcast ins(%73 : tensor<1x32x1xf16>) outs(%63 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %75 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%60 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.subf %in, %in_9 : f16
                  %116 = math.exp %115 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x512xf16>
                loom.semaphore_give %62 : memref<1x32x512xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %73 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%55 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.subf %in, %in_9 : f16
                  %116 = math.exp %115 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%37 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.mulf %in, %in_9 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x32x512xf16>) outs(%78 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %115 = arith.addf %in, %out : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x1xf16>
                %80 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%46 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %54 : memref<1x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %80 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%46 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.mulf %in, %in_9 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x128xf16>
                %c0_7 = arith.constant 0 : index
                %82 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %25, %c0_7)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%82], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_8, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %23], LR : [%arg5, %23]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %83 = loom.bufferize_to_tensor %65[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %84 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %85 = linalg.batch_matmul ins(%76, %83 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%84 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                loom.semaphore_give %65 : memref<1x512x128xf16>
                loom.semaphore_give %59 : memref<1x32x512xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %81 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%29 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.addf %in, %in_9 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %48 : memref<1x32x128xf16>
                loom.semaphore_give %45 : memref<1x32x128xf16>
                %87 = linalg.copy ins(%73 : tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                loom.semaphore_give %51 : memref<1x32x1xf16>
                loom.semaphore_give %21 : memref<1x32x128xf16>
                %88 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %90 = loom.init_tensor %89[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %87 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%90 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = math.log %in : f16
                  %116 = arith.addf %115, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %39 : memref<1x32x1xf16>
                %92 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %35 : memref<1x32x1xf16>
                %93 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %94 = loom.semaphore_take %93 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %95 = loom.init_tensor %94[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %92 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%95 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.divf %in, %in_9 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %43 : memref<1x32x128xf16>
                loom.semaphore_give %27 : memref<1x32x128xf16>
                %97 = loom.bufferize_to_memref %91 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
                %98 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %99 = loom.semaphore_take %98 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                loom.gather %97, %99 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
                loom.semaphore_give %89 : memref<1x32x1xf16>
                %100 = loom.bufferize_to_tensor %99[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %101 = loom.bufferize_to_memref %96 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %102 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %103 = loom.semaphore_take %102 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                loom.gather %101, %103 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
                loom.semaphore_give %94 : memref<1x32x128xf16>
                %104 = loom.bufferize_to_tensor %103[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %105 = arith.cmpi eq, %19, %c0 : index
                %106 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %107 = loom.semaphore_take %106 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %108 = loom.init_tensor %107[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %109 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %110 = loom.semaphore_take %109 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %111 = loom.init_tensor %110[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %112 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %113 = loom.semaphore_take %112 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %114 = loom.init_tensor %113[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                scf.if %105 {
                  %115 = linalg.fill ins(%cst_1 : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%100 : tensor<16x1x32x1xf16>) outs(%115 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %127 = arith.maximumf %in, %out : f16
                    linalg.yield %127 : f16
                  } -> tensor<1x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%100, %116 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%111 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %127 = arith.subf %in, %in_12 : f16
                    %128 = math.exp %127 : f16
                    linalg.yield %128 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %99 : memref<16x1x32x1xf16>
                  loom.semaphore_give %33 : memref<1x32x1xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%117 : tensor<16x1x32x1xf16>) outs(%118 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %127 = arith.addf %in, %out : f16
                    linalg.yield %127 : f16
                  } -> tensor<1x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117, %119 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%111 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %127 = arith.divf %in, %in_12 : f16
                    linalg.yield %127 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %31 : memref<1x32x1xf16>
                  %121 = loom.broadcast ins(%120 : tensor<16x1x32x1xf16>) outs(%114 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %110 : memref<16x1x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%104, %121 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%114 : tensor<16x1x32x128xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %127 = arith.mulf %in, %in_12 : f16
                    linalg.yield %127 : f16
                  } -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %103 : memref<16x1x32x128xf16>
                  %123 = linalg.fill ins(%cst : f16) outs(%108 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%122 : tensor<16x1x32x128xf16>) outs(%123 : tensor<1x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %127 = arith.addf %in, %out : f16
                    linalg.yield %127 : f16
                  } -> tensor<1x32x128xf16>
                  loom.semaphore_give %113 : memref<16x1x32x128xf16>
                  %c0_9 = arith.constant 0 : index
                  %c0_10 = arith.constant 0 : index
                  %125 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%125], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %126 = loom.bufferize_to_memref %124 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                  loom.copy %126, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %23], LR : [%arg5, %23]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %107 : memref<1x32x128xf16>
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
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %26)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%67], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %68 = arith.addi %arg6, %23 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %68], LR : [%arg5, %68]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %69 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %70 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %71 = linalg.batch_matmul ins(%25, %69 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%70 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              %72 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%71 : tensor<1x32x512xf16>) outs(%72 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.maximumf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73 : tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %74 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.cmpf ogt, %in, %in_9 : f16
                %118 = arith.select %117, %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x512xf16>
              %77 = loom.broadcast ins(%75 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %77 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %79 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x32x512xf16>) outs(%80 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.addf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %82 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%47 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %82 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%47 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %26, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %68], LR : [%arg5, %68]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %85 = loom.bufferize_to_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %86 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %87 = linalg.batch_matmul ins(%78, %85 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.addf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %49 : memref<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %89 = linalg.copy ins(%75 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              loom.semaphore_give %21 : memref<1x32x128xf16>
              %90 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %91 = loom.semaphore_take %90 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %92 = loom.init_tensor %91[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %89 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%92 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = math.log %in : f16
                %118 = arith.addf %117, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %94 = loom.broadcast ins(%81 : tensor<1x32x1xf16>) outs(%45 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %95 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %96 = loom.semaphore_take %95 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %97 = loom.init_tensor %96[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %94 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%97 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.divf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %99 = loom.bufferize_to_memref %93 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %100 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %101 = loom.semaphore_take %100 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              loom.gather %99, %101 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 2] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %91 : memref<1x32x1xf16>
              %102 = loom.bufferize_to_tensor %101[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %103 = loom.bufferize_to_memref %98 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %104 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %105 = loom.semaphore_take %104 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %103, %105 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 2] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %96 : memref<1x32x128xf16>
              %106 = loom.bufferize_to_tensor %105[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %107 = arith.cmpi eq, %19, %c0 : index
              %108 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %109 = loom.semaphore_take %108 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %110 = loom.init_tensor %109[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %111 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %112 = loom.semaphore_take %111 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %113 = loom.init_tensor %112[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %114 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %115 = loom.semaphore_take %114 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %116 = loom.init_tensor %115[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %107 {
                %117 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%102 : tensor<16x1x32x1xf16>) outs(%117 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.maximumf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %118 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.subf %in, %in_12 : f16
                  %131 = math.exp %130 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %101 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %120 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<16x1x32x1xf16>) outs(%120 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %121 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.divf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %32 : memref<1x32x1xf16>
                %123 = loom.broadcast ins(%122 : tensor<16x1x32x1xf16>) outs(%116 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %112 : memref<16x1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %123 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%116 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.mulf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %105 : memref<16x1x32x128xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%110 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x128xf16>) outs(%125 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %115 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %127 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%127], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %128 = loom.bufferize_to_memref %126 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %129 = arith.addi %arg6, %23 : index
                loom.copy %128, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %129], LR : [%arg5, %129]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %109 : memref<1x32x128xf16>
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
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %26)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%67], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %68 = arith.addi %arg6, %23 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %68], LR : [%arg5, %68]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %69 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %70 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %71 = linalg.batch_matmul ins(%25, %69 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%70 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              %72 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%71 : tensor<1x32x512xf16>) outs(%72 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.maximumf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73 : tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %74 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.cmpf ogt, %in, %in_9 : f16
                %118 = arith.select %117, %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x512xf16>
              %77 = loom.broadcast ins(%75 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %77 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %79 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x32x512xf16>) outs(%80 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.addf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %82 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%47 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %82 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%47 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %26, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %68], LR : [%arg5, %68]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %85 = loom.bufferize_to_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %86 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %87 = linalg.batch_matmul ins(%78, %85 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.addf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %49 : memref<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %89 = linalg.copy ins(%75 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              loom.semaphore_give %21 : memref<1x32x128xf16>
              %90 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %91 = loom.semaphore_take %90 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %92 = loom.init_tensor %91[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %89 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%92 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = math.log %in : f16
                %118 = arith.addf %117, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %94 = loom.broadcast ins(%81 : tensor<1x32x1xf16>) outs(%45 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %95 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %96 = loom.semaphore_take %95 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %97 = loom.init_tensor %96[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %94 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%97 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.divf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %99 = loom.bufferize_to_memref %93 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %100 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %101 = loom.semaphore_take %100 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              loom.gather %99, %101 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 4] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %91 : memref<1x32x1xf16>
              %102 = loom.bufferize_to_tensor %101[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %103 = loom.bufferize_to_memref %98 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %104 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %105 = loom.semaphore_take %104 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %103, %105 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [8, 4] region : (UL : [%c0, %23], LR : [%c7, %24]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %96 : memref<1x32x128xf16>
              %106 = loom.bufferize_to_tensor %105[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %107 = arith.cmpi eq, %19, %c0 : index
              %108 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %109 = loom.semaphore_take %108 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %110 = loom.init_tensor %109[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %111 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %112 = loom.semaphore_take %111 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %113 = loom.init_tensor %112[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %114 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %115 = loom.semaphore_take %114 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %116 = loom.init_tensor %115[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %107 {
                %117 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%102 : tensor<16x1x32x1xf16>) outs(%117 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.maximumf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %118 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.subf %in, %in_12 : f16
                  %131 = math.exp %130 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %101 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %120 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<16x1x32x1xf16>) outs(%120 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %121 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.divf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %32 : memref<1x32x1xf16>
                %123 = loom.broadcast ins(%122 : tensor<16x1x32x1xf16>) outs(%116 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %112 : memref<16x1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %123 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%116 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.mulf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %105 : memref<16x1x32x128xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%110 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x128xf16>) outs(%125 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %115 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %127 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%127], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %128 = loom.bufferize_to_memref %126 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %129 = arith.addi %arg6, %23 : index
                loom.copy %128, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %129], LR : [%arg5, %129]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %109 : memref<1x32x128xf16>
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
              %54 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %55 = loom.semaphore_take %54 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %56 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %57 = loom.semaphore_take %56 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %58 = loom.init_tensor %57[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %59 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %60 = loom.semaphore_take %59 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %61 = loom.init_tensor %60[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %62 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %63 = loom.semaphore_take %62 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %64 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%arg7, %c0_5, %23)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%64], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %65 = arith.muli %arg4, %c8 : index
              %66 = arith.addi %arg6, %65 : index
              loom.copy %reinterpret_cast_6, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %66], LR : [%arg5, %66]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %67 = loom.bufferize_to_tensor %55[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %68 = linalg.fill ins(%cst : f16) outs(%58 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %69 = linalg.batch_matmul ins(%22, %67 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%68 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %55 : memref<1x128x512xf16>
              %70 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%69 : tensor<1x32x512xf16>) outs(%70 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.maximumf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %72 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.cmpf ogt, %in, %in_9 : f16
                %118 = arith.select %117, %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69 : tensor<1x32x512xf16>) outs(%58 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x512xf16>
              %75 = loom.broadcast ins(%73 : tensor<1x32x1xf16>) outs(%61 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %75 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%58 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %73 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%35 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x32x512xf16>) outs(%78 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.addf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %80 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %80 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%44 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %82 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%arg7, %23, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%82], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %66], LR : [%arg5, %66]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %83 = loom.bufferize_to_tensor %63[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %84 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %85 = linalg.batch_matmul ins(%76, %83 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%84 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %63 : memref<1x512x128xf16>
              loom.semaphore_give %57 : memref<1x32x512xf16>
              %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %81 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%27 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.addf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              loom.semaphore_give %43 : memref<1x32x128xf16>
              %87 = linalg.copy ins(%73 : tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              loom.semaphore_give %20 : memref<1x32x128xf16>
              %88 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %89 = loom.semaphore_take %88 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %90 = loom.init_tensor %89[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %87 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%90 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = math.log %in : f16
                %118 = arith.addf %117, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %37 : memref<1x32x1xf16>
              %92 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%42 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %33 : memref<1x32x1xf16>
              %93 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %94 = loom.semaphore_take %93 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %95 = loom.init_tensor %94[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %92 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%95 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.divf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %41 : memref<1x32x128xf16>
              loom.semaphore_give %25 : memref<1x32x128xf16>
              %97 = loom.bufferize_to_memref %91 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %98 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %99 = loom.semaphore_take %98 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %100 = arith.muli %arg4, %c8 : index
              %101 = arith.addi %100, %c7 : index
              loom.gather %97, %99 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%18 : index), area : [8, 8] region : (UL : [%c0, %100], LR : [%c7, %101]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %89 : memref<1x32x1xf16>
              %102 = loom.bufferize_to_tensor %99[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %103 = loom.bufferize_to_memref %96 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %104 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %105 = loom.semaphore_take %104 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %103, %105 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%18 : index), area : [8, 8] region : (UL : [%c0, %100], LR : [%c7, %101]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %94 : memref<1x32x128xf16>
              %106 = loom.bufferize_to_tensor %105[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %107 = arith.cmpi eq, %18, %c0 : index
              %108 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %109 = loom.semaphore_take %108 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %110 = loom.init_tensor %109[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %111 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %112 = loom.semaphore_take %111 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %113 = loom.init_tensor %112[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %114 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %115 = loom.semaphore_take %114 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %116 = loom.init_tensor %115[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %107 {
                %117 = linalg.fill ins(%cst_1 : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%102 : tensor<16x1x32x1xf16>) outs(%117 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.maximumf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %118 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.subf %in, %in_12 : f16
                  %131 = math.exp %130 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %99 : memref<16x1x32x1xf16>
                loom.semaphore_give %31 : memref<1x32x1xf16>
                %120 = linalg.fill ins(%cst : f16) outs(%30 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<16x1x32x1xf16>) outs(%120 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %121 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.divf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %29 : memref<1x32x1xf16>
                %123 = loom.broadcast ins(%122 : tensor<16x1x32x1xf16>) outs(%116 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %112 : memref<16x1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %123 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%116 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.mulf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %105 : memref<16x1x32x128xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%110 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x128xf16>) outs(%125 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %115 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %127 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%127], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %128 = loom.bufferize_to_memref %126 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %129 = arith.addi %arg6, %100 : index
                loom.copy %128, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %129], LR : [%arg5, %129]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %109 : memref<1x32x128xf16>
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
                %56 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
                %57 = loom.semaphore_take %56 : memref<1x128x512xf16> -> memref<1x128x512xf16>
                %58 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %59 = loom.semaphore_take %58 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %60 = loom.init_tensor %59[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %61 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
                %62 = loom.semaphore_take %61 : memref<1x32x512xf16> -> memref<1x32x512xf16>
                %63 = loom.init_tensor %62[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
                %64 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
                %65 = loom.semaphore_take %64 : memref<1x512x128xf16> -> memref<1x512x128xf16>
                %c0_5 = arith.constant 0 : index
                %66 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %25)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%66], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %reinterpret_cast_6, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%23, %arg6], LR : [%23, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
                %67 = loom.bufferize_to_tensor %57[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %68 = linalg.fill ins(%cst : f16) outs(%60 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                %69 = linalg.batch_matmul ins(%24, %67 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%68 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
                loom.semaphore_give %57 : memref<1x128x512xf16>
                %70 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%69 : tensor<1x32x512xf16>) outs(%70 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %115 = arith.maximumf %in, %out : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x1xf16>
                %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x32x1xf16>) outs(%52 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %115 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x1xf16>
                %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %72 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%52 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.cmpf ogt, %in, %in_9 : f16
                  %116 = arith.select %115, %in, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69 : tensor<1x32x512xf16>) outs(%60 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %115 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x512xf16>
                %75 = loom.broadcast ins(%73 : tensor<1x32x1xf16>) outs(%63 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %75 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%60 : tensor<1x32x512xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.subf %in, %in_9 : f16
                  %116 = math.exp %115 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x512xf16>
                loom.semaphore_give %62 : memref<1x32x512xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %73 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%55 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.subf %in, %in_9 : f16
                  %116 = math.exp %115 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%37 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.mulf %in, %in_9 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x32x512xf16>) outs(%78 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %115 = arith.addf %in, %out : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x1xf16>
                %80 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%46 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %54 : memref<1x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %80 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%46 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.mulf %in, %in_9 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x128xf16>
                %c0_7 = arith.constant 0 : index
                %82 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %25, %c0_7)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%82], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_8, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%23, %arg6], LR : [%23, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %83 = loom.bufferize_to_tensor %65[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %84 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %85 = linalg.batch_matmul ins(%76, %83 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%84 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                loom.semaphore_give %65 : memref<1x512x128xf16>
                loom.semaphore_give %59 : memref<1x32x512xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %81 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%29 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.addf %in, %in_9 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %48 : memref<1x32x128xf16>
                loom.semaphore_give %45 : memref<1x32x128xf16>
                %87 = linalg.copy ins(%73 : tensor<1x32x1xf16>) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                loom.semaphore_give %51 : memref<1x32x1xf16>
                loom.semaphore_give %21 : memref<1x32x128xf16>
                %88 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<1x32x1xf16> -> memref<1x32x1xf16>
                %90 = loom.init_tensor %89[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %87 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%90 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = math.log %in : f16
                  %116 = arith.addf %115, %in_9 : f16
                  linalg.yield %116 : f16
                } -> tensor<1x32x1xf16>
                loom.semaphore_give %39 : memref<1x32x1xf16>
                %92 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
                loom.semaphore_give %35 : memref<1x32x1xf16>
                %93 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %94 = loom.semaphore_take %93 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %95 = loom.init_tensor %94[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %92 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%95 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %in_9: f16, %out: f16):
                  %115 = arith.divf %in, %in_9 : f16
                  linalg.yield %115 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %43 : memref<1x32x128xf16>
                loom.semaphore_give %27 : memref<1x32x128xf16>
                %97 = loom.bufferize_to_memref %91 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
                %98 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %99 = loom.semaphore_take %98 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                loom.gather %97, %99 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
                loom.semaphore_give %89 : memref<1x32x1xf16>
                %100 = loom.bufferize_to_tensor %99[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %101 = loom.bufferize_to_memref %96 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %102 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %103 = loom.semaphore_take %102 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                loom.gather %101, %103 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
                loom.semaphore_give %94 : memref<1x32x128xf16>
                %104 = loom.bufferize_to_tensor %103[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                %105 = arith.cmpi eq, %19, %c0 : index
                %106 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
                %107 = loom.semaphore_take %106 : memref<1x32x128xf16> -> memref<1x32x128xf16>
                %108 = loom.init_tensor %107[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
                %109 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
                %110 = loom.semaphore_take %109 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
                %111 = loom.init_tensor %110[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
                %112 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
                %113 = loom.semaphore_take %112 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
                %114 = loom.init_tensor %113[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
                scf.if %105 {
                  %115 = linalg.fill ins(%cst_1 : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%100 : tensor<16x1x32x1xf16>) outs(%115 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %127 = arith.maximumf %in, %out : f16
                    linalg.yield %127 : f16
                  } -> tensor<1x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%100, %116 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%111 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %127 = arith.subf %in, %in_12 : f16
                    %128 = math.exp %127 : f16
                    linalg.yield %128 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %99 : memref<16x1x32x1xf16>
                  loom.semaphore_give %33 : memref<1x32x1xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%117 : tensor<16x1x32x1xf16>) outs(%118 : tensor<1x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %127 = arith.addf %in, %out : f16
                    linalg.yield %127 : f16
                  } -> tensor<1x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%117, %119 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%111 : tensor<16x1x32x1xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %127 = arith.divf %in, %in_12 : f16
                    linalg.yield %127 : f16
                  } -> tensor<16x1x32x1xf16>
                  loom.semaphore_give %31 : memref<1x32x1xf16>
                  %121 = loom.broadcast ins(%120 : tensor<16x1x32x1xf16>) outs(%114 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %110 : memref<16x1x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%104, %121 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%114 : tensor<16x1x32x128xf16>) {
                  ^bb0(%in: f16, %in_12: f16, %out: f16):
                    %127 = arith.mulf %in, %in_12 : f16
                    linalg.yield %127 : f16
                  } -> tensor<16x1x32x128xf16>
                  loom.semaphore_give %103 : memref<16x1x32x128xf16>
                  %123 = linalg.fill ins(%cst : f16) outs(%108 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%122 : tensor<16x1x32x128xf16>) outs(%123 : tensor<1x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %127 = arith.addf %in, %out : f16
                    linalg.yield %127 : f16
                  } -> tensor<1x32x128xf16>
                  loom.semaphore_give %113 : memref<16x1x32x128xf16>
                  %c0_9 = arith.constant 0 : index
                  %c0_10 = arith.constant 0 : index
                  %125 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%125], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %126 = loom.bufferize_to_memref %124 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                  loom.copy %126, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%23, %arg6], LR : [%23, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %107 : memref<1x32x128xf16>
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
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %26)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%67], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %68 = arith.addi %arg5, %23 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%68, %arg6], LR : [%68, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %69 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %70 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %71 = linalg.batch_matmul ins(%25, %69 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%70 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              %72 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%71 : tensor<1x32x512xf16>) outs(%72 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.maximumf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73 : tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %74 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.cmpf ogt, %in, %in_9 : f16
                %118 = arith.select %117, %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x512xf16>
              %77 = loom.broadcast ins(%75 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %77 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %79 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x32x512xf16>) outs(%80 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.addf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %82 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%47 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %82 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%47 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %26, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%68, %arg6], LR : [%68, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %85 = loom.bufferize_to_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %86 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %87 = linalg.batch_matmul ins(%78, %85 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.addf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %49 : memref<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %89 = linalg.copy ins(%75 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              loom.semaphore_give %21 : memref<1x32x128xf16>
              %90 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %91 = loom.semaphore_take %90 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %92 = loom.init_tensor %91[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %89 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%92 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = math.log %in : f16
                %118 = arith.addf %117, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %94 = loom.broadcast ins(%81 : tensor<1x32x1xf16>) outs(%45 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %95 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %96 = loom.semaphore_take %95 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %97 = loom.init_tensor %96[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %94 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%97 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.divf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %99 = loom.bufferize_to_memref %93 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %100 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %101 = loom.semaphore_take %100 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              loom.gather %99, %101 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [2, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %91 : memref<1x32x1xf16>
              %102 = loom.bufferize_to_tensor %101[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %103 = loom.bufferize_to_memref %98 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %104 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %105 = loom.semaphore_take %104 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %103, %105 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [2, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %96 : memref<1x32x128xf16>
              %106 = loom.bufferize_to_tensor %105[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %107 = arith.cmpi eq, %19, %c0 : index
              %108 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %109 = loom.semaphore_take %108 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %110 = loom.init_tensor %109[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %111 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %112 = loom.semaphore_take %111 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %113 = loom.init_tensor %112[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %114 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %115 = loom.semaphore_take %114 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %116 = loom.init_tensor %115[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %107 {
                %117 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%102 : tensor<16x1x32x1xf16>) outs(%117 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.maximumf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %118 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.subf %in, %in_12 : f16
                  %131 = math.exp %130 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %101 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %120 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<16x1x32x1xf16>) outs(%120 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %121 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.divf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %32 : memref<1x32x1xf16>
                %123 = loom.broadcast ins(%122 : tensor<16x1x32x1xf16>) outs(%116 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %112 : memref<16x1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %123 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%116 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.mulf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %105 : memref<16x1x32x128xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%110 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x128xf16>) outs(%125 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %115 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %127 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%127], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %128 = loom.bufferize_to_memref %126 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %129 = arith.addi %arg5, %23 : index
                loom.copy %128, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%129, %arg6], LR : [%129, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %109 : memref<1x32x128xf16>
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
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%18, %c0_5, %26)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%67], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %68 = arith.addi %arg5, %23 : index
              loom.copy %reinterpret_cast_6, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%68, %arg6], LR : [%68, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %69 = loom.bufferize_to_tensor %58[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %70 = linalg.fill ins(%cst : f16) outs(%61 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %71 = linalg.batch_matmul ins(%25, %69 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%70 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %58 : memref<1x128x512xf16>
              %72 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%71 : tensor<1x32x512xf16>) outs(%72 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.maximumf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73 : tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %74 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.cmpf ogt, %in, %in_9 : f16
                %118 = arith.select %117, %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x512xf16>
              %77 = loom.broadcast ins(%75 : tensor<1x32x1xf16>) outs(%64 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %77 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%61 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %63 : memref<1x32x512xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %75 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%56 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %79 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x32x512xf16>) outs(%80 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.addf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %82 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%47 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %82 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%47 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%18, %26, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%68, %arg6], LR : [%68, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %85 = loom.bufferize_to_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %86 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %87 = linalg.batch_matmul ins(%78, %85 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %66 : memref<1x512x128xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %83 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%30 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.addf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %49 : memref<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              %89 = linalg.copy ins(%75 : tensor<1x32x1xf16>) outs(%41 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              loom.semaphore_give %21 : memref<1x32x128xf16>
              %90 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %91 = loom.semaphore_take %90 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %92 = loom.init_tensor %91[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %89 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%92 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = math.log %in : f16
                %118 = arith.addf %117, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %40 : memref<1x32x1xf16>
              %94 = loom.broadcast ins(%81 : tensor<1x32x1xf16>) outs(%45 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %36 : memref<1x32x1xf16>
              %95 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %96 = loom.semaphore_take %95 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %97 = loom.init_tensor %96[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %94 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%97 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.divf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %44 : memref<1x32x128xf16>
              loom.semaphore_give %28 : memref<1x32x128xf16>
              %99 = loom.bufferize_to_memref %93 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %100 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %101 = loom.semaphore_take %100 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              loom.gather %99, %101 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [4, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %91 : memref<1x32x1xf16>
              %102 = loom.bufferize_to_tensor %101[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %103 = loom.bufferize_to_memref %98 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %104 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %105 = loom.semaphore_take %104 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %103, %105 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%19 : index), area : [4, 8] region : (UL : [%23, %c0], LR : [%24, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %96 : memref<1x32x128xf16>
              %106 = loom.bufferize_to_tensor %105[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %107 = arith.cmpi eq, %19, %c0 : index
              %108 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %109 = loom.semaphore_take %108 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %110 = loom.init_tensor %109[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %111 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %112 = loom.semaphore_take %111 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %113 = loom.init_tensor %112[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %114 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %115 = loom.semaphore_take %114 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %116 = loom.init_tensor %115[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %107 {
                %117 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%102 : tensor<16x1x32x1xf16>) outs(%117 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.maximumf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %118 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.subf %in, %in_12 : f16
                  %131 = math.exp %130 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %101 : memref<16x1x32x1xf16>
                loom.semaphore_give %34 : memref<1x32x1xf16>
                %120 = linalg.fill ins(%cst : f16) outs(%33 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<16x1x32x1xf16>) outs(%120 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %121 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.divf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %32 : memref<1x32x1xf16>
                %123 = loom.broadcast ins(%122 : tensor<16x1x32x1xf16>) outs(%116 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %112 : memref<16x1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %123 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%116 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.mulf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %105 : memref<16x1x32x128xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%110 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x128xf16>) outs(%125 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %115 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %127 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%18, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%127], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %128 = loom.bufferize_to_memref %126 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %129 = arith.addi %arg5, %23 : index
                loom.copy %128, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%129, %arg6], LR : [%129, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %109 : memref<1x32x128xf16>
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
              %54 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %55 = loom.semaphore_take %54 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %56 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %57 = loom.semaphore_take %56 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %58 = loom.init_tensor %57[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %59 = loom.alloc [1, 32, 512] on @L1 : memref<1x32x512xf16>
              %60 = loom.semaphore_take %59 : memref<1x32x512xf16> -> memref<1x32x512xf16>
              %61 = loom.init_tensor %60[1, 32, 512] : memref<1x32x512xf16> -> tensor<1x32x512xf16>
              %62 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %63 = loom.semaphore_take %62 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_5 = arith.constant 0 : index
              %64 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%arg7, %c0_5, %23)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%64], sizes: [1, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
              %65 = arith.muli %arg4, %c8 : index
              %66 = arith.addi %arg5, %65 : index
              loom.copy %reinterpret_cast_6, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%66, %arg6], LR : [%66, %arg6]) : memref<1x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x512xf16>
              %67 = loom.bufferize_to_tensor %55[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
              %68 = linalg.fill ins(%cst : f16) outs(%58 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              %69 = linalg.batch_matmul ins(%22, %67 : tensor<1x32x128xf16>, tensor<1x128x512xf16>) outs(%68 : tensor<1x32x512xf16>) -> tensor<1x32x512xf16>
              loom.semaphore_give %55 : memref<1x128x512xf16>
              %70 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%69 : tensor<1x32x512xf16>) outs(%70 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.maximumf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71 : tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %72 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.cmpf ogt, %in, %in_9 : f16
                %118 = arith.select %117, %in, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%69 : tensor<1x32x512xf16>) outs(%58 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.mulf %in, %cst_2 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x512xf16>
              %75 = loom.broadcast ins(%73 : tensor<1x32x1xf16>) outs(%61 : tensor<1x32x512xf16>) dim(2) -> tensor<1x32x512xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %75 : tensor<1x32x512xf16>, tensor<1x32x512xf16>) outs(%58 : tensor<1x32x512xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x512xf16>
              loom.semaphore_give %60 : memref<1x32x512xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %73 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %77 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%35 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x32x512xf16>) outs(%78 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.addf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %80 = loom.broadcast ins(%77 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %80 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%44 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %82 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%arg7, %23, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%82], sizes: [1, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%66, %arg6], LR : [%66, %arg6]) : memref<1x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %83 = loom.bufferize_to_tensor %63[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %84 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %85 = linalg.batch_matmul ins(%76, %83 : tensor<1x32x512xf16>, tensor<1x512x128xf16>) outs(%84 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %63 : memref<1x512x128xf16>
              loom.semaphore_give %57 : memref<1x32x512xf16>
              %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %81 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%27 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.addf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %46 : memref<1x32x128xf16>
              loom.semaphore_give %43 : memref<1x32x128xf16>
              %87 = linalg.copy ins(%73 : tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %49 : memref<1x32x1xf16>
              loom.semaphore_give %20 : memref<1x32x128xf16>
              %88 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
              %89 = loom.semaphore_take %88 : memref<1x32x1xf16> -> memref<1x32x1xf16>
              %90 = loom.init_tensor %89[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
              %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %87 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%90 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = math.log %in : f16
                %118 = arith.addf %117, %in_9 : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %37 : memref<1x32x1xf16>
              %92 = loom.broadcast ins(%79 : tensor<1x32x1xf16>) outs(%42 : tensor<1x32x128xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %33 : memref<1x32x1xf16>
              %93 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %94 = loom.semaphore_take %93 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %95 = loom.init_tensor %94[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %92 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%95 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.divf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %41 : memref<1x32x128xf16>
              loom.semaphore_give %25 : memref<1x32x128xf16>
              %97 = loom.bufferize_to_memref %91 : tensor<1x32x1xf16> -> memref<1x32x1xf16>
              %98 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %99 = loom.semaphore_take %98 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %100 = arith.muli %arg4, %c8 : index
              %101 = arith.addi %100, %c7 : index
              loom.gather %97, %99 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%18 : index), area : [8, 8] region : (UL : [%100, %c0], LR : [%101, %c7]) : memref<1x32x1xf16> to memref<16x1x32x1xf16>
              loom.semaphore_give %89 : memref<1x32x1xf16>
              %102 = loom.bufferize_to_tensor %99[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %103 = loom.bufferize_to_memref %96 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              %104 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %105 = loom.semaphore_take %104 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              loom.gather %103, %105 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%18 : index), area : [8, 8] region : (UL : [%100, %c0], LR : [%101, %c7]) : memref<1x32x128xf16> to memref<16x1x32x128xf16>
              loom.semaphore_give %94 : memref<1x32x128xf16>
              %106 = loom.bufferize_to_tensor %105[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              %107 = arith.cmpi eq, %18, %c0 : index
              %108 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
              %109 = loom.semaphore_take %108 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %110 = loom.init_tensor %109[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %111 = loom.alloc [16, 1, 32, 1] on @L1 : memref<16x1x32x1xf16>
              %112 = loom.semaphore_take %111 : memref<16x1x32x1xf16> -> memref<16x1x32x1xf16>
              %113 = loom.init_tensor %112[16, 1, 32, 1] : memref<16x1x32x1xf16> -> tensor<16x1x32x1xf16>
              %114 = loom.alloc [16, 1, 32, 128] on @L1 : memref<16x1x32x128xf16>
              %115 = loom.semaphore_take %114 : memref<16x1x32x128xf16> -> memref<16x1x32x128xf16>
              %116 = loom.init_tensor %115[16, 1, 32, 128] : memref<16x1x32x128xf16> -> tensor<16x1x32x128xf16>
              scf.if %107 {
                %117 = linalg.fill ins(%cst_1 : f16) outs(%32 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%102 : tensor<16x1x32x1xf16>) outs(%117 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.maximumf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %118 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.subf %in, %in_12 : f16
                  %131 = math.exp %130 : f16
                  linalg.yield %131 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %99 : memref<16x1x32x1xf16>
                loom.semaphore_give %31 : memref<1x32x1xf16>
                %120 = linalg.fill ins(%cst : f16) outs(%30 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
                %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<16x1x32x1xf16>) outs(%120 : tensor<1x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x1xf16>
                %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%119, %121 : tensor<16x1x32x1xf16>, tensor<1x32x1xf16>) outs(%113 : tensor<16x1x32x1xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.divf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x1xf16>
                loom.semaphore_give %29 : memref<1x32x1xf16>
                %123 = loom.broadcast ins(%122 : tensor<16x1x32x1xf16>) outs(%116 : tensor<16x1x32x128xf16>) dim(3) -> tensor<16x1x32x128xf16>
                loom.semaphore_give %112 : memref<16x1x32x1xf16>
                %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%106, %123 : tensor<16x1x32x128xf16>, tensor<16x1x32x128xf16>) outs(%116 : tensor<16x1x32x128xf16>) {
                ^bb0(%in: f16, %in_12: f16, %out: f16):
                  %130 = arith.mulf %in, %in_12 : f16
                  linalg.yield %130 : f16
                } -> tensor<16x1x32x128xf16>
                loom.semaphore_give %105 : memref<16x1x32x128xf16>
                %125 = linalg.fill ins(%cst : f16) outs(%110 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%124 : tensor<16x1x32x128xf16>) outs(%125 : tensor<1x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %130 = arith.addf %in, %out : f16
                  linalg.yield %130 : f16
                } -> tensor<1x32x128xf16>
                loom.semaphore_give %115 : memref<16x1x32x128xf16>
                %c0_9 = arith.constant 0 : index
                %c0_10 = arith.constant 0 : index
                %127 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%arg7, %c0_9, %c0_10)
                %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%127], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %128 = loom.bufferize_to_memref %126 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
                %129 = arith.addi %arg5, %100 : index
                loom.copy %128, %reinterpret_cast_11 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%129, %arg6], LR : [%129, %arg6]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %109 : memref<1x32x128xf16>
              }
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
