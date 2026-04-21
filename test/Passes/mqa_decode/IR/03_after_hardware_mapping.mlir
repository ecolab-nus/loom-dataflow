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
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
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
              %34 = loom.init_tensor %33[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %35 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %36 = loom.subview %arg3[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
              %37 = loom.bufferize_to_tensor %35[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %38 = arith.muli %28, %21 : index
              %39 = arith.ceildivui %21, %22 : index
              %40 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %41 = loom.semaphore_take %40 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %42 = loom.init_tensor %41[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %43 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
              %44 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %45 = loom.semaphore_take %44 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %46 = loom.init_tensor %45[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %47 = loom.semaphore_take %44 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %48 = loom.init_tensor %47[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %49 = loom.semaphore_take %44 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %50 = loom.init_tensor %49[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %51 = linalg.fill ins(%cst_0 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
              %52 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %53 = loom.semaphore_take %52 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %54 = loom.init_tensor %53[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %55 = loom.semaphore_take %52 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %56 = loom.init_tensor %55[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %57 = linalg.fill ins(%cst_1 : f16) outs(%56 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
              %58 = loom.alloc [%20, 32, 128] on @L1 : memref<?x32x128xf16>
              %59 = loom.semaphore_take %58 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %60 = loom.init_tensor %59[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %61 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %63 = loom.init_tensor %62[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %64 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %65 = loom.semaphore_take %64 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %66 = loom.init_tensor %65[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %67 = loom.alloc [%20, 32, 1] on @L1 : memref<?x32x1xf16>
              %68 = loom.semaphore_take %67 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %69 = loom.init_tensor %68[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %70 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
              %71 = loom.semaphore_take %70 : memref<?x128x?xf16> -> memref<?x128x?xf16>
              %72 = loom.alloc [%20, 32, %22] on @L1 : memref<?x32x?xf16>
              %73 = loom.semaphore_take %72 : memref<?x32x?xf16> -> memref<?x32x?xf16>
              %74 = loom.init_tensor %73[%20, 32, %22] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
              %75 = loom.alloc [%20, 32, 32] on @L1 : memref<?x32x32xf16>
              %76 = loom.semaphore_take %75 : memref<?x32x32xf16> -> memref<?x32x32xf16>
              %77 = loom.init_tensor %76[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
              %78 = loom.semaphore_take %75 : memref<?x32x32xf16> -> memref<?x32x32xf16>
              %79 = loom.init_tensor %78[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
              %80 = loom.semaphore_take %75 : memref<?x32x32xf16> -> memref<?x32x32xf16>
              %81 = loom.init_tensor %80[%20, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
              %82 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
              %83 = loom.semaphore_take %82 : memref<?x?x128xf16> -> memref<?x?x128xf16>
              %84:3 = scf.for %arg8 = %c0 to %39 step %c1 iter_args(%arg9 = %57, %arg10 = %51, %arg11 = %43) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                %106 = arith.muli %arg8, %22 : index
                %107 = arith.addi %38, %106 : index
                %108 = loom.subview %arg0[%29, 0, %107] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                loom.copy %108, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                %109 = loom.bufferize_to_tensor %71[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %110 = linalg.fill ins(%cst : f16) outs(%74 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                %111 = linalg.batch_matmul ins(%37, %109 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%110 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                loom.semaphore_give %71 : memref<?x128x?xf16>
                %112 = linalg.fill ins(%cst_1 : f16) outs(%63 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%111 : tensor<?x32x?xf16>) outs(%112 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %128 = arith.maximumf %in, %out : f16
                  linalg.yield %128 : f16
                } -> tensor<?x32x1xf16>
                %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %113 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %128 = arith.mulf %in_3, %cst_2 : f16
                  %129 = arith.cmpf ogt, %in, %128 : f16
                  %130 = arith.select %129, %in, %128 : f16
                  linalg.yield %130 : f16
                } -> tensor<?x32x1xf16>
                %115 = loom.broadcast ins(%114 : tensor<?x32x1xf16>) outs(%81 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%111, %115 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%74 : tensor<?x32x?xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %128 = arith.mulf %in, %cst_2 : f16
                  %129 = arith.subf %128, %in_3 : f16
                  %130 = math.exp %129 : f16
                  linalg.yield %130 : f16
                } -> tensor<?x32x?xf16>
                loom.semaphore_give %80 : memref<?x32x32xf16>
                %117 = linalg.fill ins(%cst : f16) outs(%66 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %128 = arith.addf %in, %out : f16
                  linalg.yield %128 : f16
                } -> tensor<?x32x1xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %114 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%69 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %128 = arith.subf %in, %in_3 : f16
                  %129 = math.exp %128 : f16
                  linalg.yield %129 : f16
                } -> tensor<?x32x1xf16>
                %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                  %128 = arith.mulf %in, %in_3 : f16
                  %129 = arith.addf %128, %in_4 : f16
                  linalg.yield %129 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %65 : memref<?x32x1xf16>
                %121 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%79 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %68 : memref<?x32x1xf16>
                %122 = loom.subview %arg1[%29, %107, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                loom.copy %122, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %123 = loom.bufferize_to_tensor %83[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %124 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %125 = linalg.batch_matmul ins(%116, %123 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%124 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                loom.semaphore_give %83 : memref<?x?x128xf16>
                loom.semaphore_give %73 : memref<?x32x?xf16>
                %126 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%125, %arg11, %121 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg11 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                  %128 = arith.mulf %in_3, %in_4 : f16
                  %129 = arith.addf %in, %128 : f16
                  linalg.yield %129 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %78 : memref<?x32x32xf16>
                loom.semaphore_give %59 : memref<?x32x128xf16>
                %127 = linalg.copy ins(%114 : tensor<?x32x1xf16>) outs(%arg9 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                loom.semaphore_give %62 : memref<?x32x1xf16>
                scf.yield %127, %120, %126 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              loom.semaphore_give %35 : memref<?x32x128xf16>
              %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84#1, %84#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%54 : tensor<?x32x1xf16>) {
              ^bb0(%in: f16, %in_3: f16, %out: f16):
                %106 = math.log %in : f16
                %107 = arith.addf %106, %in_3 : f16
                linalg.yield %107 : f16
              } -> tensor<?x32x1xf16>
              loom.semaphore_give %55 : memref<?x32x1xf16>
              %86 = loom.broadcast ins(%84#1 : tensor<?x32x1xf16>) outs(%77 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
              loom.semaphore_give %49 : memref<?x32x1xf16>
              %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84#2, %86 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%34 : tensor<?x32x128xf16>) {
              ^bb0(%in: f16, %in_3: f16, %out: f16):
                %106 = arith.divf %in, %in_3 : f16
                linalg.yield %106 : f16
              } -> tensor<?x32x128xf16>
              loom.semaphore_give %76 : memref<?x32x32xf16>
              loom.semaphore_give %41 : memref<?x32x128xf16>
              %88 = loom.alloc [%24, %20, 32, 1] on @L1 : memref<?x?x32x1xf16>
              %89 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
              %90 = loom.init_tensor %89[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
              %91 = loom.semaphore_take %52 : memref<?x32x1xf16> -> memref<?x32x1xf16>
              %92 = loom.init_tensor %91[%20, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
              %93 = loom.sync ins(%85 : tensor<?x32x1xf16>) outs(%92 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
              %94 = loom.gather ins(%93 : tensor<?x32x1xf16>) outs(%90 : tensor<?x?x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<?x?x32x1xf16>
              loom.semaphore_give %91 : memref<?x32x1xf16>
              loom.semaphore_give %53 : memref<?x32x1xf16>
              %95 = loom.alloc [%24, %20, 32, 128] on @L1 : memref<?x?x32x128xf16>
              %96 = loom.semaphore_take %95 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
              %97 = loom.init_tensor %96[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
              %98 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
              %99 = loom.init_tensor %98[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
              %100 = loom.sync ins(%87 : tensor<?x32x128xf16>) outs(%99 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
              %101 = loom.gather ins(%100 : tensor<?x32x128xf16>) outs(%97 : tensor<?x?x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<?x?x32x128xf16>
              loom.semaphore_give %98 : memref<?x32x128xf16>
              loom.semaphore_give %33 : memref<?x32x128xf16>
              %102 = loom.alloc [%24, %20, 32, 32] on @L1 : memref<?x?x32x32xf16>
              %103 = loom.semaphore_take %102 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
              %104 = loom.init_tensor %103[%24, %20, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
              %105 = arith.cmpi eq, %arg5, %c0 : index
              scf.if %105 {
                %106 = linalg.fill ins(%cst_1 : f16) outs(%48 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %107 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %108 = loom.init_tensor %107[%24, %20, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %109 = loom.sync ins(%94 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x?x32x1xf16>) -> tensor<?x?x32x1xf16>
                %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%109 : tensor<?x?x32x1xf16>) outs(%106 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %127 = arith.maximumf %in, %out : f16
                  linalg.yield %127 : f16
                } -> tensor<?x32x1xf16>
                %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%90 : tensor<?x?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %127 = arith.subf %in, %in_3 : f16
                  %128 = math.exp %127 : f16
                  linalg.yield %128 : f16
                } -> tensor<?x?x32x1xf16>
                loom.semaphore_give %47 : memref<?x32x1xf16>
                %112 = linalg.fill ins(%cst : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %127 = arith.addf %in, %out : f16
                  linalg.yield %127 : f16
                } -> tensor<?x32x1xf16>
                %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%90 : tensor<?x?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %127 = arith.divf %in, %in_3 : f16
                  linalg.yield %127 : f16
                } -> tensor<?x?x32x1xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%104 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                loom.semaphore_give %107 : memref<?x?x32x1xf16>
                loom.semaphore_give %89 : memref<?x?x32x1xf16>
                %116 = loom.semaphore_take %95 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %117 = loom.init_tensor %116[%24, %20, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %118 = loom.sync ins(%101 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x?x32x128xf16>) -> tensor<?x?x32x128xf16>
                %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%118, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%97 : tensor<?x?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %127 = arith.mulf %in, %in_3 : f16
                  linalg.yield %127 : f16
                } -> tensor<?x?x32x128xf16>
                loom.semaphore_give %103 : memref<?x?x32x32xf16>
                %120 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%119 : tensor<?x?x32x128xf16>) outs(%120 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %127 = arith.addf %in, %out : f16
                  linalg.yield %127 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %116 : memref<?x?x32x128xf16>
                loom.semaphore_give %96 : memref<?x?x32x128xf16>
                %122 = loom.subview %arg2[%29, 0, 0] [%20, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %123 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %124 = loom.init_tensor %123[%20, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %125 = loom.sync ins(%121 : tensor<?x32x128xf16>) outs(%124 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %126 = loom.bufferize_to_memref %125 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                loom.copy %126, %122 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.semaphore_give %123 : memref<?x32x128xf16>
                loom.semaphore_give %31 : memref<?x32x128xf16>
              }
            } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
