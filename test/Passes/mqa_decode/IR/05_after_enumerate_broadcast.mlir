module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y1y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
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
      %18 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %19 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %21 = arith.ceildivui %c16, %18 : index
      %22 = arith.ceildivui %c8192, %19 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (1) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg7 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg8 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                %27 = arith.muli %25, %18 : index
                %28 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %30 = loom.subview %arg3[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %31 = arith.addi %arg6, %arg4 : index
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %31], LR : [%c7, %31]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %32 = loom.bufferize_to_tensor %29[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %33 = arith.muli %26, %19 : index
                %34 = arith.ceildivui %19, %20 : index
                %35 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %36 = loom.semaphore_take %35 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %37 = loom.init_tensor %36[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %39 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %41 = loom.init_tensor %40[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %42 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %43 = loom.init_tensor %42[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %44 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = linalg.fill ins(%cst_0 : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %47 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %53 = loom.init_tensor %52[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %54 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %56 = loom.init_tensor %55[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %57 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %59 = loom.init_tensor %58[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %60 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %62 = loom.init_tensor %61[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %63 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %64 = loom.semaphore_take %63 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %65 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %67 = loom.init_tensor %66[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %68 = loom.alloc [%18, 32, 32] on @L1 : memref<?x32x32xf16>
                %69 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %70 = loom.init_tensor %69[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %71 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %72 = loom.init_tensor %71[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %73 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %74 = loom.init_tensor %73[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %75 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %76 = loom.semaphore_take %75 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %77:3 = scf.for %arg9 = %c0 to %34 step %c1 iter_args(%arg10 = %50, %arg11 = %46, %arg12 = %38) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %108 = arith.muli %arg9, %20 : index
                  %109 = arith.addi %33, %108 : index
                  %110 = loom.subview %arg0[%27, 0, %109] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %110, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %111 = loom.bufferize_to_tensor %64[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%67 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %113 = linalg.batch_matmul ins(%32, %111 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%112 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %64 : memref<?x128x?xf16>
                  %114 = linalg.fill ins(%cst_1 : f16) outs(%56 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%113 : tensor<?x32x?xf16>) outs(%114 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %130 = arith.maximumf %in, %out : f16
                    linalg.yield %130 : f16
                  } -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %115 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%56 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %130 = arith.mulf %in_3, %cst_2 : f16
                    %131 = arith.cmpf ogt, %in, %130 : f16
                    %132 = arith.select %131, %in, %130 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = loom.broadcast ins(%116 : tensor<?x32x1xf16>) outs(%74 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%113, %117 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%67 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %130 = arith.mulf %in, %cst_2 : f16
                    %131 = arith.subf %130, %in_3 : f16
                    %132 = math.exp %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %73 : memref<?x32x32xf16>
                  %119 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%118 : tensor<?x32x?xf16>) outs(%119 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %130 = arith.addf %in, %out : f16
                    linalg.yield %130 : f16
                  } -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %116 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%62 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %130 = arith.subf %in, %in_3 : f16
                    %131 = math.exp %130 : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %121, %120 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %130 = arith.mulf %in, %in_3 : f16
                    %131 = arith.addf %130, %in_4 : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %58 : memref<?x32x1xf16>
                  %123 = loom.broadcast ins(%121 : tensor<?x32x1xf16>) outs(%72 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %61 : memref<?x32x1xf16>
                  %124 = loom.subview %arg1[%27, %109, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %124, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %125 = loom.bufferize_to_tensor %76[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %126 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %127 = linalg.batch_matmul ins(%118, %125 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%126 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %76 : memref<?x?x128xf16>
                  loom.semaphore_give %66 : memref<?x32x?xf16>
                  %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127, %arg12, %123 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %130 = arith.mulf %in_3, %in_4 : f16
                    %131 = arith.addf %in, %130 : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %71 : memref<?x32x32xf16>
                  loom.semaphore_give %52 : memref<?x32x128xf16>
                  %129 = linalg.copy ins(%116 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %55 : memref<?x32x1xf16>
                  scf.yield %129, %122, %128 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %29 : memref<?x32x128xf16>
                %78 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %79 = loom.semaphore_take %78 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %80 = loom.init_tensor %79[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77#1, %77#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%80 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %108 = math.log %in : f16
                  %109 = arith.addf %108, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %82 = loom.broadcast ins(%77#1 : tensor<?x32x1xf16>) outs(%70 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %44 : memref<?x32x1xf16>
                %83 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %84 = loom.semaphore_take %83 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %85 = loom.init_tensor %84[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77#2, %82 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%85 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %108 = arith.divf %in, %in_3 : f16
                  linalg.yield %108 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %69 : memref<?x32x32xf16>
                loom.semaphore_give %36 : memref<?x32x128xf16>
                %87 = loom.bufferize_to_memref %81 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %88 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %87, %89 across(%26 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %79 : memref<?x32x1xf16>
                %90 = loom.bufferize_to_tensor %89[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %91 = loom.bufferize_to_memref %86 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %92 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %93 = loom.semaphore_take %92 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %91, %93 across(%26 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %84 : memref<?x32x128xf16>
                %94 = loom.bufferize_to_tensor %93[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %95 = arith.cmpi eq, %26, %c0 : index
                %96 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %98 = loom.init_tensor %97[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %99 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %100 = loom.semaphore_take %99 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %101 = loom.init_tensor %100[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %102 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %104 = loom.init_tensor %103[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %105 = loom.alloc [%22, %18, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %107 = loom.init_tensor %106[%22, %18, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                scf.if %95 {
                  %108 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%90 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.maximumf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%90, %109 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.subf %in, %in_3 : f16
                    %121 = math.exp %120 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %89 : memref<?x?x32x1xf16>
                  loom.semaphore_give %42 : memref<?x32x1xf16>
                  %111 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%110 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.addf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %112 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.divf %in, %in_3 : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %40 : memref<?x32x1xf16>
                  %114 = loom.broadcast ins(%113 : tensor<?x?x32x1xf16>) outs(%107 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %100 : memref<?x?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %114 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%104 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.mulf %in, %in_3 : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x32xf16>
                  loom.semaphore_give %93 : memref<?x?x32x128xf16>
                  %116 = linalg.fill ins(%cst : f16) outs(%98 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%115 : tensor<?x?x32x128xf16>) outs(%116 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.addf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x128xf16>
                  %118 = loom.subview %arg2[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %119 = loom.bufferize_to_memref %117 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %97 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y2y4__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc2_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %18 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %19 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %21 = arith.ceildivui %c16, %18 : index
      %22 = arith.ceildivui %c8192, %19 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (2) {
            %23 = arith.ceildivui %21, %c4 : index
            scf.for %arg7 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c16 : index
              scf.for %arg8 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg5, %arg6, %arg8)
                %27 = arith.muli %25, %18 : index
                %28 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %30 = loom.subview %arg3[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %31 = arith.muli %arg4, %c2 : index
                %32 = arith.addi %31, %c1 : index
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %31], LR : [%c7, %32]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %29[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %26, %19 : index
                %35 = arith.ceildivui %19, %20 : index
                %36 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%18, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %109 = arith.muli %arg9, %20 : index
                  %110 = arith.addi %34, %109 : index
                  %111 = loom.subview %arg0[%27, 0, %110] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %112 = arith.addi %arg6, %31 : index
                  loom.copy %111, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %112], LR : [%arg5, %112]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %113 = loom.bufferize_to_tensor %65[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %115 = linalg.batch_matmul ins(%33, %113 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%114 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  %116 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%115 : tensor<?x32x?xf16>) outs(%116 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.maximumf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in_3, %cst_2 : f16
                    %133 = arith.cmpf ogt, %in, %132 : f16
                    %134 = arith.select %133, %in, %132 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = loom.broadcast ins(%118 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %119 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in, %cst_2 : f16
                    %133 = arith.subf %132, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %121 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%120 : tensor<?x32x?xf16>) outs(%121 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.addf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.subf %in, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %123, %122 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in, %in_3 : f16
                    %133 = arith.addf %132, %in_4 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %125 = loom.broadcast ins(%123 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %126 = loom.subview %arg1[%27, %110, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %126, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %112], LR : [%arg5, %112]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %127 = loom.bufferize_to_tensor %77[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %129 = linalg.batch_matmul ins(%120, %127 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%128 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %arg12, %125 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in_3, %in_4 : f16
                    %133 = arith.addf %in, %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %131 = linalg.copy ins(%118 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %131, %124, %130 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %29 : memref<?x32x128xf16>
                %79 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = math.log %in : f16
                  %110 = arith.addf %109, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = arith.divf %in, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.bufferize_to_memref %82 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %89 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %90 = loom.semaphore_take %89 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %88, %90 across(%26 : index), area : [8, 2] region : (UL : [%c0, %31], LR : [%c7, %32]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %91 = loom.bufferize_to_tensor %90[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %92 = loom.bufferize_to_memref %87 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %93 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %92, %94 across(%26 : index), area : [8, 2] region : (UL : [%c0, %31], LR : [%c7, %32]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %95 = loom.bufferize_to_tensor %94[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %96 = arith.cmpi eq, %26, %c0 : index
                %97 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %98 = loom.semaphore_take %97 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %99 = loom.init_tensor %98[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %100 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %101 = loom.semaphore_take %100 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %102 = loom.init_tensor %101[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %103 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %104 = loom.semaphore_take %103 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %105 = loom.init_tensor %104[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %106 = loom.alloc [%22, %18, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %107 = loom.semaphore_take %106 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %108 = loom.init_tensor %107[%22, %18, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                scf.if %96 {
                  %109 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%91 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.maximumf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.subf %in, %in_3 : f16
                    %123 = math.exp %122 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %90 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.divf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %101 : memref<?x?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%105 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.mulf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x32xf16>
                  loom.semaphore_give %94 : memref<?x?x32x128xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%99 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%116 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x128xf16>
                  %119 = loom.subview %arg2[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %121 = arith.addi %arg6, %31 : index
                  loom.copy %120, %119 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %121], LR : [%arg5, %121]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %98 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y4y2__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc4_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %18 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %19 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %21 = arith.ceildivui %c16, %18 : index
      %22 = arith.ceildivui %c8192, %19 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (4) {
            %23 = arith.ceildivui %21, %c2 : index
            scf.for %arg7 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c32 : index
              scf.for %arg8 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg5, %arg6, %arg8)
                %27 = arith.muli %25, %18 : index
                %28 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %30 = loom.subview %arg3[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %31 = arith.muli %arg4, %c4 : index
                %32 = arith.addi %31, %c3 : index
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %31], LR : [%c7, %32]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %29[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %26, %19 : index
                %35 = arith.ceildivui %19, %20 : index
                %36 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%18, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %109 = arith.muli %arg9, %20 : index
                  %110 = arith.addi %34, %109 : index
                  %111 = loom.subview %arg0[%27, 0, %110] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %112 = arith.addi %arg6, %31 : index
                  loom.copy %111, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %112], LR : [%arg5, %112]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %113 = loom.bufferize_to_tensor %65[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %115 = linalg.batch_matmul ins(%33, %113 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%114 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  %116 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%115 : tensor<?x32x?xf16>) outs(%116 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.maximumf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in_3, %cst_2 : f16
                    %133 = arith.cmpf ogt, %in, %132 : f16
                    %134 = arith.select %133, %in, %132 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = loom.broadcast ins(%118 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %119 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in, %cst_2 : f16
                    %133 = arith.subf %132, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %121 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%120 : tensor<?x32x?xf16>) outs(%121 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.addf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.subf %in, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %123, %122 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in, %in_3 : f16
                    %133 = arith.addf %132, %in_4 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %125 = loom.broadcast ins(%123 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %126 = loom.subview %arg1[%27, %110, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %126, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %112], LR : [%arg5, %112]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %127 = loom.bufferize_to_tensor %77[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %129 = linalg.batch_matmul ins(%120, %127 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%128 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %arg12, %125 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in_3, %in_4 : f16
                    %133 = arith.addf %in, %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %131 = linalg.copy ins(%118 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %131, %124, %130 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %29 : memref<?x32x128xf16>
                %79 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = math.log %in : f16
                  %110 = arith.addf %109, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = arith.divf %in, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.bufferize_to_memref %82 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %89 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %90 = loom.semaphore_take %89 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %88, %90 across(%26 : index), area : [8, 4] region : (UL : [%c0, %31], LR : [%c7, %32]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %91 = loom.bufferize_to_tensor %90[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %92 = loom.bufferize_to_memref %87 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %93 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %92, %94 across(%26 : index), area : [8, 4] region : (UL : [%c0, %31], LR : [%c7, %32]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %95 = loom.bufferize_to_tensor %94[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %96 = arith.cmpi eq, %26, %c0 : index
                %97 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %98 = loom.semaphore_take %97 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %99 = loom.init_tensor %98[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %100 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %101 = loom.semaphore_take %100 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %102 = loom.init_tensor %101[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %103 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %104 = loom.semaphore_take %103 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %105 = loom.init_tensor %104[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %106 = loom.alloc [%22, %18, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %107 = loom.semaphore_take %106 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %108 = loom.init_tensor %107[%22, %18, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                scf.if %96 {
                  %109 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%91 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.maximumf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.subf %in, %in_3 : f16
                    %123 = math.exp %122 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %90 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.divf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %101 : memref<?x?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%105 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.mulf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x32xf16>
                  loom.semaphore_give %94 : memref<?x?x32x128xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%99 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%116 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x128xf16>
                  %119 = loom.subview %arg2[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %121 = arith.addi %arg6, %31 : index
                  loom.copy %120, %119 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %121], LR : [%arg5, %121]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %98 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y8y1__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %18 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %19 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %21 = arith.ceildivui %c16, %18 : index
      %22 = arith.ceildivui %c8192, %19 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %21 step %c1 {
              %23 = arith.ceildivui %22, %c64 : index
              scf.for %arg8 = %c0 to %23 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %25 = arith.muli %arg7, %18 : index
                %26 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %27 = loom.semaphore_take %26 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %28 = loom.subview %arg3[%25, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %28, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %29 = loom.bufferize_to_tensor %27[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %30 = arith.muli %24, %19 : index
                %31 = arith.ceildivui %19, %20 : index
                %32 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %33 = loom.semaphore_take %32 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %34 = loom.init_tensor %33[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %35 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %36 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %38 = loom.init_tensor %37[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %39 = loom.semaphore_take %36 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %40 = loom.init_tensor %39[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %41 = loom.semaphore_take %36 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %44 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %45 = loom.semaphore_take %44 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %50 = loom.init_tensor %49[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %51 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %56 = loom.init_tensor %55[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %57 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %59 = loom.init_tensor %58[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %60 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %62 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %64 = loom.init_tensor %63[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %65 = loom.alloc [%18, 32, 32] on @L1 : memref<?x32x32xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %67 = loom.init_tensor %66[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %68 = loom.semaphore_take %65 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %69 = loom.init_tensor %68[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %70 = loom.semaphore_take %65 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %74:3 = scf.for %arg9 = %c0 to %31 step %c1 iter_args(%arg10 = %47, %arg11 = %43, %arg12 = %35) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %107 = arith.muli %arg9, %20 : index
                  %108 = arith.addi %30, %107 : index
                  %109 = loom.subview %arg0[%25, 0, %108] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %110 = arith.muli %arg4, %c8 : index
                  %111 = arith.addi %arg6, %110 : index
                  loom.copy %109, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %111], LR : [%arg5, %111]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %112 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %114 = linalg.batch_matmul ins(%29, %112 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%113 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %115 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%114 : tensor<?x32x?xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.maximumf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %116 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%53 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in_3, %cst_2 : f16
                    %132 = arith.cmpf ogt, %in, %131 : f16
                    %133 = arith.select %132, %in, %131 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%114, %118 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%64 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in, %cst_2 : f16
                    %132 = arith.subf %131, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %70 : memref<?x32x32xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.addf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%59 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.subf %in, %in_3 : f16
                    %132 = math.exp %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %122, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in, %in_3 : f16
                    %132 = arith.addf %131, %in_4 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %55 : memref<?x32x1xf16>
                  %124 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%69 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %58 : memref<?x32x1xf16>
                  %125 = loom.subview %arg1[%25, %108, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %125, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %111], LR : [%arg5, %111]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %126 = loom.bufferize_to_tensor %73[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %127 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %128 = linalg.batch_matmul ins(%119, %126 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%127 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %73 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x32x?xf16>
                  %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %arg12, %124 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in_3, %in_4 : f16
                    %132 = arith.addf %in, %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %68 : memref<?x32x32xf16>
                  loom.semaphore_give %49 : memref<?x32x128xf16>
                  %130 = linalg.copy ins(%117 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %52 : memref<?x32x1xf16>
                  scf.yield %130, %123, %129 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %27 : memref<?x32x128xf16>
                %75 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %76 = loom.semaphore_take %75 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %77 = loom.init_tensor %76[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74#1, %74#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%77 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %107 = math.log %in : f16
                  %108 = arith.addf %107, %in_3 : f16
                  linalg.yield %108 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %79 = loom.broadcast ins(%74#1 : tensor<?x32x1xf16>) outs(%67 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %41 : memref<?x32x1xf16>
                %80 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %81 = loom.semaphore_take %80 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %82 = loom.init_tensor %81[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74#2, %79 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%82 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %107 = arith.divf %in, %in_3 : f16
                  linalg.yield %107 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %66 : memref<?x32x32xf16>
                loom.semaphore_give %33 : memref<?x32x128xf16>
                %84 = loom.bufferize_to_memref %78 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %85 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %86 = loom.semaphore_take %85 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %87 = arith.muli %arg4, %c8 : index
                %88 = arith.addi %87, %c7 : index
                loom.gather %84, %86 across(%24 : index), area : [8, 8] region : (UL : [%c0, %87], LR : [%c7, %88]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %76 : memref<?x32x1xf16>
                %89 = loom.bufferize_to_tensor %86[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %90 = loom.bufferize_to_memref %83 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %91 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %92 = loom.semaphore_take %91 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %90, %92 across(%24 : index), area : [8, 8] region : (UL : [%c0, %87], LR : [%c7, %88]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %81 : memref<?x32x128xf16>
                %93 = loom.bufferize_to_tensor %92[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %94 = arith.cmpi eq, %24, %c0 : index
                %95 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %96 = loom.semaphore_take %95 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %97 = loom.init_tensor %96[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %98 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %99 = loom.semaphore_take %98 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %100 = loom.init_tensor %99[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %101 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %102 = loom.semaphore_take %101 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %103 = loom.init_tensor %102[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %104 = loom.alloc [%22, %18, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %106 = loom.init_tensor %105[%22, %18, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                scf.if %94 {
                  %107 = linalg.fill ins(%cst_1 : f16) outs(%40 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%89 : tensor<?x?x32x1xf16>) outs(%107 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.maximumf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x1xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89, %108 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%100 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.subf %in, %in_3 : f16
                    %121 = math.exp %120 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %86 : memref<?x?x32x1xf16>
                  loom.semaphore_give %39 : memref<?x32x1xf16>
                  %110 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%109 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.addf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %111 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%100 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.divf %in, %in_3 : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %37 : memref<?x32x1xf16>
                  %113 = loom.broadcast ins(%112 : tensor<?x?x32x1xf16>) outs(%106 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %99 : memref<?x?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %113 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%103 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.mulf %in, %in_3 : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x32xf16>
                  loom.semaphore_give %92 : memref<?x?x32x128xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%97 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%114 : tensor<?x?x32x128xf16>) outs(%115 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.addf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %102 : memref<?x?x32x128xf16>
                  %117 = loom.subview %arg2[%25, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %118 = loom.bufferize_to_memref %116 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %119 = arith.addi %arg6, %87 : index
                  loom.copy %118, %117 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %119], LR : [%arg5, %119]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %96 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x1x8_y8__d0i1_d1i1_d2i0__f01__dim_y_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
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
      %18 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %19 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %21 = arith.ceildivui %c16, %18 : index
      %22 = arith.ceildivui %c8192, %19 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (1) {
          affine.parallel (%arg6) = (0) to (8) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg7 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg8 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg6, %arg8)
                %27 = arith.muli %25, %18 : index
                %28 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %30 = loom.subview %arg3[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %31 = arith.addi %arg5, %arg4 : index
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%31, %c0], LR : [%31, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %32 = loom.bufferize_to_tensor %29[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %33 = arith.muli %26, %19 : index
                %34 = arith.ceildivui %19, %20 : index
                %35 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %36 = loom.semaphore_take %35 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %37 = loom.init_tensor %36[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %39 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %41 = loom.init_tensor %40[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %42 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %43 = loom.init_tensor %42[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %44 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = linalg.fill ins(%cst_0 : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %47 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %53 = loom.init_tensor %52[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %54 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %56 = loom.init_tensor %55[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %57 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %59 = loom.init_tensor %58[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %60 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %62 = loom.init_tensor %61[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %63 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %64 = loom.semaphore_take %63 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %65 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %67 = loom.init_tensor %66[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %68 = loom.alloc [%18, 32, 32] on @L1 : memref<?x32x32xf16>
                %69 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %70 = loom.init_tensor %69[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %71 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %72 = loom.init_tensor %71[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %73 = loom.semaphore_take %68 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %74 = loom.init_tensor %73[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %75 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %76 = loom.semaphore_take %75 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %77:3 = scf.for %arg9 = %c0 to %34 step %c1 iter_args(%arg10 = %50, %arg11 = %46, %arg12 = %38) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %108 = arith.muli %arg9, %20 : index
                  %109 = arith.addi %33, %108 : index
                  %110 = loom.subview %arg0[%27, 0, %109] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %110, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%31, %arg6], LR : [%31, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %111 = loom.bufferize_to_tensor %64[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%67 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %113 = linalg.batch_matmul ins(%32, %111 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%112 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %64 : memref<?x128x?xf16>
                  %114 = linalg.fill ins(%cst_1 : f16) outs(%56 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%113 : tensor<?x32x?xf16>) outs(%114 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %130 = arith.maximumf %in, %out : f16
                    linalg.yield %130 : f16
                  } -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %115 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%56 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %130 = arith.mulf %in_3, %cst_2 : f16
                    %131 = arith.cmpf ogt, %in, %130 : f16
                    %132 = arith.select %131, %in, %130 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = loom.broadcast ins(%116 : tensor<?x32x1xf16>) outs(%74 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%113, %117 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%67 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %130 = arith.mulf %in, %cst_2 : f16
                    %131 = arith.subf %130, %in_3 : f16
                    %132 = math.exp %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %73 : memref<?x32x32xf16>
                  %119 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%118 : tensor<?x32x?xf16>) outs(%119 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %130 = arith.addf %in, %out : f16
                    linalg.yield %130 : f16
                  } -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %116 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%62 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %130 = arith.subf %in, %in_3 : f16
                    %131 = math.exp %130 : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %121, %120 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %130 = arith.mulf %in, %in_3 : f16
                    %131 = arith.addf %130, %in_4 : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %58 : memref<?x32x1xf16>
                  %123 = loom.broadcast ins(%121 : tensor<?x32x1xf16>) outs(%72 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %61 : memref<?x32x1xf16>
                  %124 = loom.subview %arg1[%27, %109, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %124, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%31, %arg6], LR : [%31, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %125 = loom.bufferize_to_tensor %76[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %126 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %127 = linalg.batch_matmul ins(%118, %125 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%126 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %76 : memref<?x?x128xf16>
                  loom.semaphore_give %66 : memref<?x32x?xf16>
                  %128 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127, %arg12, %123 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %130 = arith.mulf %in_3, %in_4 : f16
                    %131 = arith.addf %in, %130 : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %71 : memref<?x32x32xf16>
                  loom.semaphore_give %52 : memref<?x32x128xf16>
                  %129 = linalg.copy ins(%116 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %55 : memref<?x32x1xf16>
                  scf.yield %129, %122, %128 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %29 : memref<?x32x128xf16>
                %78 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %79 = loom.semaphore_take %78 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %80 = loom.init_tensor %79[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77#1, %77#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%80 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %108 = math.log %in : f16
                  %109 = arith.addf %108, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %82 = loom.broadcast ins(%77#1 : tensor<?x32x1xf16>) outs(%70 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %44 : memref<?x32x1xf16>
                %83 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %84 = loom.semaphore_take %83 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %85 = loom.init_tensor %84[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77#2, %82 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%85 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %108 = arith.divf %in, %in_3 : f16
                  linalg.yield %108 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %69 : memref<?x32x32xf16>
                loom.semaphore_give %36 : memref<?x32x128xf16>
                %87 = loom.bufferize_to_memref %81 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %88 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %89 = loom.semaphore_take %88 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %87, %89 across(%26 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %79 : memref<?x32x1xf16>
                %90 = loom.bufferize_to_tensor %89[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %91 = loom.bufferize_to_memref %86 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %92 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %93 = loom.semaphore_take %92 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %91, %93 across(%26 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %84 : memref<?x32x128xf16>
                %94 = loom.bufferize_to_tensor %93[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %95 = arith.cmpi eq, %26, %c0 : index
                %96 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %98 = loom.init_tensor %97[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %99 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %100 = loom.semaphore_take %99 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %101 = loom.init_tensor %100[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %102 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %104 = loom.init_tensor %103[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %105 = loom.alloc [%22, %18, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %107 = loom.init_tensor %106[%22, %18, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                scf.if %95 {
                  %108 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%90 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.maximumf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%90, %109 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.subf %in, %in_3 : f16
                    %121 = math.exp %120 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %89 : memref<?x?x32x1xf16>
                  loom.semaphore_give %42 : memref<?x32x1xf16>
                  %111 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%110 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.addf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %112 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%101 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.divf %in, %in_3 : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %40 : memref<?x32x1xf16>
                  %114 = loom.broadcast ins(%113 : tensor<?x?x32x1xf16>) outs(%107 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %100 : memref<?x?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %114 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%104 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.mulf %in, %in_3 : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x32xf16>
                  loom.semaphore_give %93 : memref<?x?x32x128xf16>
                  %116 = linalg.fill ins(%cst : f16) outs(%98 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%115 : tensor<?x?x32x128xf16>) outs(%116 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.addf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x128xf16>
                  %118 = loom.subview %arg2[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %119 = loom.bufferize_to_memref %117 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%31, %arg6], LR : [%31, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %97 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x2x4_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc2_dim_y_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %18 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %19 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %21 = arith.ceildivui %c16, %18 : index
      %22 = arith.ceildivui %c8192, %19 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (2) {
          affine.parallel (%arg6) = (0) to (8) {
            %23 = arith.ceildivui %21, %c4 : index
            scf.for %arg7 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c16 : index
              scf.for %arg8 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg5, %arg6, %arg8)
                %27 = arith.muli %25, %18 : index
                %28 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %30 = loom.subview %arg3[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %31 = arith.muli %arg4, %c2 : index
                %32 = arith.addi %31, %c1 : index
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%31, %c0], LR : [%32, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %29[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %26, %19 : index
                %35 = arith.ceildivui %19, %20 : index
                %36 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%18, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %109 = arith.muli %arg9, %20 : index
                  %110 = arith.addi %34, %109 : index
                  %111 = loom.subview %arg0[%27, 0, %110] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %112 = arith.addi %arg5, %31 : index
                  loom.copy %111, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%112, %arg6], LR : [%112, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %113 = loom.bufferize_to_tensor %65[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %115 = linalg.batch_matmul ins(%33, %113 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%114 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  %116 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%115 : tensor<?x32x?xf16>) outs(%116 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.maximumf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in_3, %cst_2 : f16
                    %133 = arith.cmpf ogt, %in, %132 : f16
                    %134 = arith.select %133, %in, %132 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = loom.broadcast ins(%118 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %119 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in, %cst_2 : f16
                    %133 = arith.subf %132, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %121 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%120 : tensor<?x32x?xf16>) outs(%121 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.addf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.subf %in, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %123, %122 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in, %in_3 : f16
                    %133 = arith.addf %132, %in_4 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %125 = loom.broadcast ins(%123 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %126 = loom.subview %arg1[%27, %110, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %126, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%112, %arg6], LR : [%112, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %127 = loom.bufferize_to_tensor %77[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %129 = linalg.batch_matmul ins(%120, %127 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%128 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %arg12, %125 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in_3, %in_4 : f16
                    %133 = arith.addf %in, %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %131 = linalg.copy ins(%118 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %131, %124, %130 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %29 : memref<?x32x128xf16>
                %79 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = math.log %in : f16
                  %110 = arith.addf %109, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = arith.divf %in, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.bufferize_to_memref %82 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %89 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %90 = loom.semaphore_take %89 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %88, %90 across(%26 : index), area : [2, 8] region : (UL : [%31, %c0], LR : [%32, %c7]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %91 = loom.bufferize_to_tensor %90[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %92 = loom.bufferize_to_memref %87 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %93 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %92, %94 across(%26 : index), area : [2, 8] region : (UL : [%31, %c0], LR : [%32, %c7]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %95 = loom.bufferize_to_tensor %94[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %96 = arith.cmpi eq, %26, %c0 : index
                %97 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %98 = loom.semaphore_take %97 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %99 = loom.init_tensor %98[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %100 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %101 = loom.semaphore_take %100 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %102 = loom.init_tensor %101[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %103 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %104 = loom.semaphore_take %103 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %105 = loom.init_tensor %104[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %106 = loom.alloc [%22, %18, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %107 = loom.semaphore_take %106 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %108 = loom.init_tensor %107[%22, %18, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                scf.if %96 {
                  %109 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%91 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.maximumf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.subf %in, %in_3 : f16
                    %123 = math.exp %122 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %90 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.divf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %101 : memref<?x?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%105 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.mulf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x32xf16>
                  loom.semaphore_give %94 : memref<?x?x32x128xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%99 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%116 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x128xf16>
                  %119 = loom.subview %arg2[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %121 = arith.addi %arg5, %31 : index
                  loom.copy %120, %119 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%121, %arg6], LR : [%121, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %98 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x4x2_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc4_dim_y_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %18 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %19 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %21 = arith.ceildivui %c16, %18 : index
      %22 = arith.ceildivui %c8192, %19 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (4) {
          affine.parallel (%arg6) = (0) to (8) {
            %23 = arith.ceildivui %21, %c2 : index
            scf.for %arg7 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c32 : index
              scf.for %arg8 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg5, %arg6, %arg8)
                %27 = arith.muli %25, %18 : index
                %28 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %29 = loom.semaphore_take %28 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %30 = loom.subview %arg3[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %31 = arith.muli %arg4, %c4 : index
                %32 = arith.addi %31, %c3 : index
                loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%31, %c0], LR : [%32, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %33 = loom.bufferize_to_tensor %29[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %34 = arith.muli %26, %19 : index
                %35 = arith.ceildivui %19, %20 : index
                %36 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %38 = loom.init_tensor %37[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %40 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %50 = loom.init_tensor %49[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %51 = linalg.fill ins(%cst_1 : f16) outs(%50 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %52 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %54 = loom.init_tensor %53[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %55 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %57 = loom.init_tensor %56[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %58 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %60 = loom.init_tensor %59[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %61 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %63 = loom.init_tensor %62[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %64 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %66 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %68 = loom.init_tensor %67[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %69 = loom.alloc [%18, 32, 32] on @L1 : memref<?x32x32xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %73 = loom.init_tensor %72[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %74 = loom.semaphore_take %69 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %75 = loom.init_tensor %74[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %76 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %51, %arg11 = %47, %arg12 = %39) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %109 = arith.muli %arg9, %20 : index
                  %110 = arith.addi %34, %109 : index
                  %111 = loom.subview %arg0[%27, 0, %110] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %112 = arith.addi %arg5, %31 : index
                  loom.copy %111, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%112, %arg6], LR : [%112, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %113 = loom.bufferize_to_tensor %65[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %115 = linalg.batch_matmul ins(%33, %113 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%114 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %65 : memref<?x128x?xf16>
                  %116 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%115 : tensor<?x32x?xf16>) outs(%116 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.maximumf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%57 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in_3, %cst_2 : f16
                    %133 = arith.cmpf ogt, %in, %132 : f16
                    %134 = arith.select %133, %in, %132 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = loom.broadcast ins(%118 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %119 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%68 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in, %cst_2 : f16
                    %133 = arith.subf %132, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x32xf16>
                  %121 = linalg.fill ins(%cst : f16) outs(%60 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%120 : tensor<?x32x?xf16>) outs(%121 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.addf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%63 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.subf %in, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %123, %122 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in, %in_3 : f16
                    %133 = arith.addf %132, %in_4 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %59 : memref<?x32x1xf16>
                  %125 = loom.broadcast ins(%123 : tensor<?x32x1xf16>) outs(%73 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %62 : memref<?x32x1xf16>
                  %126 = loom.subview %arg1[%27, %110, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %126, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%112, %arg6], LR : [%112, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %127 = loom.bufferize_to_tensor %77[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %129 = linalg.batch_matmul ins(%120, %127 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%128 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x32x?xf16>
                  %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %arg12, %125 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in_3, %in_4 : f16
                    %133 = arith.addf %in, %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %72 : memref<?x32x32xf16>
                  loom.semaphore_give %53 : memref<?x32x128xf16>
                  %131 = linalg.copy ins(%118 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %56 : memref<?x32x1xf16>
                  scf.yield %131, %124, %130 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %29 : memref<?x32x128xf16>
                %79 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = math.log %in : f16
                  %110 = arith.addf %109, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %49 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %84 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = arith.divf %in, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %70 : memref<?x32x32xf16>
                loom.semaphore_give %37 : memref<?x32x128xf16>
                %88 = loom.bufferize_to_memref %82 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %89 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %90 = loom.semaphore_take %89 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %88, %90 across(%26 : index), area : [4, 8] region : (UL : [%31, %c0], LR : [%32, %c7]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %91 = loom.bufferize_to_tensor %90[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %92 = loom.bufferize_to_memref %87 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %93 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %92, %94 across(%26 : index), area : [4, 8] region : (UL : [%31, %c0], LR : [%32, %c7]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %95 = loom.bufferize_to_tensor %94[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %96 = arith.cmpi eq, %26, %c0 : index
                %97 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %98 = loom.semaphore_take %97 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %99 = loom.init_tensor %98[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %100 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %101 = loom.semaphore_take %100 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %102 = loom.init_tensor %101[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %103 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %104 = loom.semaphore_take %103 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %105 = loom.init_tensor %104[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %106 = loom.alloc [%22, %18, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %107 = loom.semaphore_take %106 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %108 = loom.init_tensor %107[%22, %18, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                scf.if %96 {
                  %109 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%91 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.maximumf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.subf %in, %in_3 : f16
                    %123 = math.exp %122 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %90 : memref<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%102 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.divf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %41 : memref<?x32x1xf16>
                  %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %101 : memref<?x?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%105 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %122 = arith.mulf %in, %in_3 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x32xf16>
                  loom.semaphore_give %94 : memref<?x?x32x128xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%99 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%116 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %122 = arith.addf %in, %out : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x128xf16>
                  %119 = loom.subview %arg2[%27, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %121 = arith.addi %arg5, %31 : index
                  loom.copy %120, %119 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%121, %arg6], LR : [%121, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %98 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8x1_y8__d0i1_d1i1_d2i0__f01__dim_x_level1_bc8_dim_y_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %18 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %19 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 64 : index} : index
      %21 = arith.ceildivui %c16, %18 : index
      %22 = arith.ceildivui %c8192, %19 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %21 step %c1 {
              %23 = arith.ceildivui %22, %c64 : index
              scf.for %arg8 = %c0 to %23 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %25 = arith.muli %arg7, %18 : index
                %26 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %27 = loom.semaphore_take %26 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %28 = loom.subview %arg3[%25, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %28, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %29 = loom.bufferize_to_tensor %27[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %30 = arith.muli %24, %19 : index
                %31 = arith.ceildivui %19, %20 : index
                %32 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %33 = loom.semaphore_take %32 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %34 = loom.init_tensor %33[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %35 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %36 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %37 = loom.semaphore_take %36 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %38 = loom.init_tensor %37[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %39 = loom.semaphore_take %36 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %40 = loom.init_tensor %39[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %41 = loom.semaphore_take %36 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %42 = loom.init_tensor %41[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %44 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %45 = loom.semaphore_take %44 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %48 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %49 = loom.semaphore_take %48 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %50 = loom.init_tensor %49[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %51 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %56 = loom.init_tensor %55[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %57 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %59 = loom.init_tensor %58[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %60 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %62 = loom.alloc [%18, 32, %20] on @L1 : memref<?x32x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %64 = loom.init_tensor %63[%18, 32, %20] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %65 = loom.alloc [%18, 32, 32] on @L1 : memref<?x32x32xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %67 = loom.init_tensor %66[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %68 = loom.semaphore_take %65 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %69 = loom.init_tensor %68[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %70 = loom.semaphore_take %65 : memref<?x32x32xf16> -> memref<?x32x32xf16>
                %71 = loom.init_tensor %70[%18, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
                %72 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %74:3 = scf.for %arg9 = %c0 to %31 step %c1 iter_args(%arg10 = %47, %arg11 = %43, %arg12 = %35) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %107 = arith.muli %arg9, %20 : index
                  %108 = arith.addi %30, %107 : index
                  %109 = loom.subview %arg0[%25, 0, %108] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %110 = arith.muli %arg4, %c8 : index
                  %111 = arith.addi %arg5, %110 : index
                  loom.copy %109, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%111, %arg6], LR : [%111, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %112 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %114 = linalg.batch_matmul ins(%29, %112 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%113 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %115 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%114 : tensor<?x32x?xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.maximumf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %116 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%53 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in_3, %cst_2 : f16
                    %132 = arith.cmpf ogt, %in, %131 : f16
                    %133 = arith.select %132, %in, %131 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x32x1xf16>) outs(%71 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%114, %118 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%64 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in, %cst_2 : f16
                    %132 = arith.subf %131, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %70 : memref<?x32x32xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.addf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%59 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.subf %in, %in_3 : f16
                    %132 = math.exp %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %122, %121 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in, %in_3 : f16
                    %132 = arith.addf %131, %in_4 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %55 : memref<?x32x1xf16>
                  %124 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%69 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %58 : memref<?x32x1xf16>
                  %125 = loom.subview %arg1[%25, %108, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %125, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%111, %arg6], LR : [%111, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %126 = loom.bufferize_to_tensor %73[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %127 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %128 = linalg.batch_matmul ins(%119, %126 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%127 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %73 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x32x?xf16>
                  %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %arg12, %124 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in_3, %in_4 : f16
                    %132 = arith.addf %in, %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %68 : memref<?x32x32xf16>
                  loom.semaphore_give %49 : memref<?x32x128xf16>
                  %130 = linalg.copy ins(%117 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %52 : memref<?x32x1xf16>
                  scf.yield %130, %123, %129 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %27 : memref<?x32x128xf16>
                %75 = loom.alloc [%18, 32, 1] on @L1 : memref<?x32x1xf16>
                %76 = loom.semaphore_take %75 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %77 = loom.init_tensor %76[%18, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74#1, %74#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%77 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %107 = math.log %in : f16
                  %108 = arith.addf %107, %in_3 : f16
                  linalg.yield %108 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %45 : memref<?x32x1xf16>
                %79 = loom.broadcast ins(%74#1 : tensor<?x32x1xf16>) outs(%67 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %41 : memref<?x32x1xf16>
                %80 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %81 = loom.semaphore_take %80 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %82 = loom.init_tensor %81[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74#2, %79 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%82 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %107 = arith.divf %in, %in_3 : f16
                  linalg.yield %107 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %66 : memref<?x32x32xf16>
                loom.semaphore_give %33 : memref<?x32x128xf16>
                %84 = loom.bufferize_to_memref %78 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %85 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %86 = loom.semaphore_take %85 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %87 = arith.muli %arg4, %c8 : index
                %88 = arith.addi %87, %c7 : index
                loom.gather %84, %86 across(%24 : index), area : [8, 8] region : (UL : [%87, %c0], LR : [%88, %c7]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %76 : memref<?x32x1xf16>
                %89 = loom.bufferize_to_tensor %86[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %90 = loom.bufferize_to_memref %83 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %91 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %92 = loom.semaphore_take %91 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %90, %92 across(%24 : index), area : [8, 8] region : (UL : [%87, %c0], LR : [%88, %c7]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %81 : memref<?x32x128xf16>
                %93 = loom.bufferize_to_tensor %92[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %94 = arith.cmpi eq, %24, %c0 : index
                %95 = loom.alloc [%18, 32, 128] on @L1 : memref<?x32x128xf16>
                %96 = loom.semaphore_take %95 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %97 = loom.init_tensor %96[%18, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %98 = loom.alloc [%22, %18, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %99 = loom.semaphore_take %98 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %100 = loom.init_tensor %99[%22, %18, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %101 = loom.alloc [%22, %18, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %102 = loom.semaphore_take %101 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %103 = loom.init_tensor %102[%22, %18, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %104 = loom.alloc [%22, %18, 32, 32] on @L1 : memref<?x?x32x32xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
                %106 = loom.init_tensor %105[%22, %18, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
                scf.if %94 {
                  %107 = linalg.fill ins(%cst_1 : f16) outs(%40 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%89 : tensor<?x?x32x1xf16>) outs(%107 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.maximumf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x1xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89, %108 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%100 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.subf %in, %in_3 : f16
                    %121 = math.exp %120 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %86 : memref<?x?x32x1xf16>
                  loom.semaphore_give %39 : memref<?x32x1xf16>
                  %110 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%109 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.addf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%109, %111 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%100 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.divf %in, %in_3 : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %37 : memref<?x32x1xf16>
                  %113 = loom.broadcast ins(%112 : tensor<?x?x32x1xf16>) outs(%106 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %99 : memref<?x?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %113 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%103 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %120 = arith.mulf %in, %in_3 : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x32xf16>
                  loom.semaphore_give %92 : memref<?x?x32x128xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%97 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%114 : tensor<?x?x32x128xf16>) outs(%115 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %120 = arith.addf %in, %out : f16
                    linalg.yield %120 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %102 : memref<?x?x32x128xf16>
                  %117 = loom.subview %arg2[%25, 0, 0] [%18, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %118 = loom.bufferize_to_memref %116 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %119 = arith.addi %arg5, %87 : index
                  loom.copy %118, %117 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%119, %arg6], LR : [%119, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %96 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
