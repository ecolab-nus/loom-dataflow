module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.memory.bank "mem_bank", {bsize = 16 : i64, nblk = 5464 : i64}
  %4 = adl.spatial_dim "dim_nbank", 16
  %5 = adl.memory.array "mem_L1", [%4] of %3
  %6 = adl.resource.exclusive "res_matrix_lane"
  %7 = adl.resource.exclusive "res_vector_lane"
  %8 = adl.processor.compute @proc_matrix_lane, from %5 to %5, with [%6]
  %9 = adl.processor.compute @proc_vector_lane, from %5 to %5, with [%7]
  %10 = adl.arch.compose "arch_mesh", arch[%8, %9], mem[%5]
  %11 = adl.spatial_dim "dim_x", 8
  %12 = adl.spatial_dim "dim_y", 8
  %13 = adl.memory.array "mem_array_L1", [%11, %12] of %5
  %14 = adl.arch.scale "arch_mesh", [%11, %12] of %10, mem_region %13
  %15 = adl.resource.exclusive "res_noc0"
  %16 = adl.resource.exclusive "res_noc1"
  %17 = adl.processor.dmover @proc_dram_l1_noc0, from %2 to %13, with [%15]
  %18 = adl.processor.dmover @proc_l1_l1_noc0, from %13 to %13, with [%15]
  %19 = adl.processor.dmover @proc_l1_dram_noc1, from %13 to %2, with [%16]
  %20 = adl.arch.compose "arch_system", arch[%14, %17, %18, %19], mem[%2]
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y1y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c16 = arith.constant 16 : index
      %c8192 = arith.constant 8192 : index
      %21 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %22 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_n {upper_bound = 8192 : index} : index
      %24 = arith.ceildivui %c16, %21 : index
      %25 = arith.ceildivui %c8192, %22 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (1) {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %25, %c8 : index
              scf.for %arg8 = %c0 to %27 step %c1 {
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %33 = loom.subview %arg2[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %34 = arith.addi %arg6, %arg4 : index
                loom.copy %33, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %34], LR : [%c7, %34]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %35 = loom.bufferize_to_tensor %32[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %36 = arith.muli %29, %22 : index
                %37 = arith.ceildivui %22, %23 : index
                %38 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %39 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %40 = loom.init_tensor %39[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %42 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %48 = loom.init_tensor %47[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %49 = linalg.fill ins(%cst_0 : f16) outs(%48 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %50 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %51 = loom.semaphore_take %50 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %52 = loom.init_tensor %51[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %53 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %54 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %56 = loom.init_tensor %55[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %57 = loom.semaphore_take %54 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %58 = loom.init_tensor %57[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %59 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %60 = loom.semaphore_take %59 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %61 = loom.init_tensor %60[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %62 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %63 = loom.semaphore_take %62 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %64 = loom.init_tensor %63[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %65 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %67 = loom.init_tensor %66[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %68 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %69 = loom.semaphore_take %68 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %70 = loom.init_tensor %69[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %71 = loom.alloc [%21, 128, %23] on @L1 : memref<?x128x?xf16>
                %72 = loom.semaphore_take %71 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %73 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %74 = loom.semaphore_take %73 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %75 = loom.init_tensor %74[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %76 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %77 = loom.semaphore_take %76 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %78 = loom.init_tensor %77[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %79 = loom.alloc [%21, %23, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81:3 = scf.for %arg9 = %c0 to %37 step %c1 iter_args(%arg10 = %53, %arg11 = %49, %arg12 = %41) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %109 = arith.muli %arg9, %23 : index
                  %110 = arith.addi %36, %109 : index
                  %111 = loom.subview %arg0[%30, 0, %110] [%21, 128, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %111, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %34], LR : [%arg5, %34]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %112 = loom.bufferize_to_tensor %72[%21, 128, %23] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%75 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %114 = linalg.batch_matmul ins(%35, %112 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%113 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %72 : memref<?x128x?xf16>
                  %115 = linalg.fill ins(%cst_1 : f16) outs(%64 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%114 : tensor<?x32x?xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.maximumf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %116 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%64 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in_3, %cst_2 : f16
                    %132 = arith.cmpf ogt, %in, %131 : f16
                    %133 = arith.select %132, %in, %131 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x32x1xf16>) outs(%78 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%114, %118 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%75 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in, %cst_2 : f16
                    %132 = arith.subf %131, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %77 : memref<?x32x?xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%67 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.addf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%70 : tensor<?x32x1xf16>) {
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
                  loom.semaphore_give %66 : memref<?x32x1xf16>
                  %124 = loom.subview %arg1[%30, %110, 0] [%21, %23, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %124, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %34], LR : [%arg5, %34]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %125 = loom.bufferize_to_tensor %80[%21, %23, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %126 = linalg.fill ins(%cst : f16) outs(%58 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %127 = linalg.batch_matmul ins(%119, %125 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%126 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %74 : memref<?x32x?xf16>
                  %128 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%61 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %69 : memref<?x32x1xf16>
                  %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127, %arg12, %128 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in_3, %in_4 : f16
                    %132 = arith.addf %in, %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %60 : memref<?x32x128xf16>
                  loom.semaphore_give %57 : memref<?x32x128xf16>
                  %130 = linalg.copy ins(%117 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %63 : memref<?x32x1xf16>
                  scf.yield %130, %123, %129 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %32 : memref<?x32x128xf16>
                %82 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %83 = loom.semaphore_take %82 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %84 = loom.init_tensor %83[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#1, %81#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%84 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = math.log %in : f16
                  %110 = arith.addf %109, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %51 : memref<?x32x1xf16>
                %86 = loom.broadcast ins(%81#1 : tensor<?x32x1xf16>) outs(%56 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %47 : memref<?x32x1xf16>
                %87 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %88 = loom.semaphore_take %87 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %89 = loom.init_tensor %88[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#2, %86 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%89 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = arith.divf %in, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %55 : memref<?x32x128xf16>
                loom.semaphore_give %39 : memref<?x32x128xf16>
                %91 = loom.bufferize_to_memref %85 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %92 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %93 = loom.semaphore_take %92 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %91, %93 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %83 : memref<?x32x1xf16>
                %94 = loom.bufferize_to_tensor %93[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %95 = loom.bufferize_to_memref %90 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %96 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %95, %97 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %88 : memref<?x32x128xf16>
                %98 = loom.bufferize_to_tensor %97[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %99 = arith.cmpi eq, %29, %c0 : index
                %100 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %101 = loom.semaphore_take %100 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %102 = loom.init_tensor %101[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %103 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %104 = loom.semaphore_take %103 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %105 = loom.init_tensor %104[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %106 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %107 = loom.semaphore_take %106 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %108 = loom.init_tensor %107[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                scf.if %99 {
                  %109 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%94 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.maximumf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%105 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.subf %in, %in_3 : f16
                    %122 = math.exp %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %93 : memref<?x?x32x1xf16>
                  loom.semaphore_give %45 : memref<?x32x1xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%105 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.divf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%108 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.mulf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %97 : memref<?x?x32x128xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%102 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%116 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x128xf16>
                  %119 = loom.subview %arg3[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %120, %119 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %34], LR : [%arg5, %34]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %101 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
      %c16 = arith.constant 16 : index
      %c8192 = arith.constant 8192 : index
      %21 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %22 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_n {upper_bound = 8192 : index} : index
      %24 = arith.ceildivui %c16, %21 : index
      %25 = arith.ceildivui %c8192, %22 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (2) {
            %26 = arith.ceildivui %24, %c4 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %25, %c16 : index
              scf.for %arg8 = %c0 to %27 step %c1 {
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %29 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg5, %arg6, %arg8)
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %33 = loom.subview %arg2[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %34 = arith.muli %arg4, %c2 : index
                %35 = arith.addi %34, %c1 : index
                loom.copy %33, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %34], LR : [%c7, %35]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %36 = loom.bufferize_to_tensor %32[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %37 = arith.muli %29, %22 : index
                %38 = arith.ceildivui %22, %23 : index
                %39 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %41 = loom.init_tensor %40[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %42 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %43 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %47 = loom.init_tensor %46[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %48 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %55 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %57 = loom.init_tensor %56[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %58 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %59 = loom.init_tensor %58[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %60 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %62 = loom.init_tensor %61[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %63 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %64 = loom.semaphore_take %63 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %65 = loom.init_tensor %64[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %66 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %68 = loom.init_tensor %67[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %69 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %71 = loom.init_tensor %70[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %72 = loom.alloc [%21, 128, %23] on @L1 : memref<?x128x?xf16>
                %73 = loom.semaphore_take %72 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %74 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %75 = loom.semaphore_take %74 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %76 = loom.init_tensor %75[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %77 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %78 = loom.semaphore_take %77 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %79 = loom.init_tensor %78[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %80 = loom.alloc [%21, %23, 128] on @L1 : memref<?x?x128xf16>
                %81 = loom.semaphore_take %80 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %82:3 = scf.for %arg9 = %c0 to %38 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %42) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %110 = arith.muli %arg9, %23 : index
                  %111 = arith.addi %37, %110 : index
                  %112 = loom.subview %arg0[%30, 0, %111] [%21, 128, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %113 = arith.addi %arg6, %34 : index
                  loom.copy %112, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %113], LR : [%arg5, %113]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %114 = loom.bufferize_to_tensor %73[%21, 128, %23] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%76 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %116 = linalg.batch_matmul ins(%36, %114 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%115 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %73 : memref<?x128x?xf16>
                  %117 = linalg.fill ins(%cst_1 : f16) outs(%65 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%65 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in_3, %cst_2 : f16
                    %134 = arith.cmpf ogt, %in, %133 : f16
                    %135 = arith.select %134, %in, %133 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%79 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %120 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%76 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in, %cst_2 : f16
                    %134 = arith.subf %133, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %78 : memref<?x32x?xf16>
                  %122 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121 : tensor<?x32x?xf16>) outs(%122 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%71 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.subf %in, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %124, %123 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in, %in_3 : f16
                    %134 = arith.addf %133, %in_4 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %67 : memref<?x32x1xf16>
                  %126 = loom.subview %arg1[%30, %111, 0] [%21, %23, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %126, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %113], LR : [%arg5, %113]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %127 = loom.bufferize_to_tensor %81[%21, %23, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %129 = linalg.batch_matmul ins(%121, %127 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%128 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %81 : memref<?x?x128xf16>
                  loom.semaphore_give %75 : memref<?x32x?xf16>
                  %130 = loom.broadcast ins(%124 : tensor<?x32x1xf16>) outs(%62 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %70 : memref<?x32x1xf16>
                  %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %arg12, %130 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in_3, %in_4 : f16
                    %134 = arith.addf %in, %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %61 : memref<?x32x128xf16>
                  loom.semaphore_give %58 : memref<?x32x128xf16>
                  %132 = linalg.copy ins(%119 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %64 : memref<?x32x1xf16>
                  scf.yield %132, %125, %131 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %32 : memref<?x32x128xf16>
                %83 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %84 = loom.semaphore_take %83 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %85 = loom.init_tensor %84[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82#1, %82#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%85 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = math.log %in : f16
                  %111 = arith.addf %110, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %52 : memref<?x32x1xf16>
                %87 = loom.broadcast ins(%82#1 : tensor<?x32x1xf16>) outs(%57 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %88 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %89 = loom.semaphore_take %88 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %90 = loom.init_tensor %89[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82#2, %87 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%90 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = arith.divf %in, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %56 : memref<?x32x128xf16>
                loom.semaphore_give %40 : memref<?x32x128xf16>
                %92 = loom.bufferize_to_memref %86 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %93 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %92, %94 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [8, 2] region : (UL : [%c0, %34], LR : [%c7, %35]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %84 : memref<?x32x1xf16>
                %95 = loom.bufferize_to_tensor %94[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %96 = loom.bufferize_to_memref %91 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %97 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %98 = loom.semaphore_take %97 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %96, %98 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [8, 2] region : (UL : [%c0, %34], LR : [%c7, %35]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %89 : memref<?x32x128xf16>
                %99 = loom.bufferize_to_tensor %98[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %100 = arith.cmpi eq, %29, %c0 : index
                %101 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %102 = loom.semaphore_take %101 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %103 = loom.init_tensor %102[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %104 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %106 = loom.init_tensor %105[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %107 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %108 = loom.semaphore_take %107 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %109 = loom.init_tensor %108[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                scf.if %100 {
                  %110 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%95 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.maximumf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %111 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.subf %in, %in_3 : f16
                    %124 = math.exp %123 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %94 : memref<?x?x32x1xf16>
                  loom.semaphore_give %46 : memref<?x32x1xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%112 : tensor<?x?x32x1xf16>) outs(%113 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %114 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.divf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %44 : memref<?x32x1xf16>
                  %116 = loom.broadcast ins(%115 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %116 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%109 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.mulf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %98 : memref<?x?x32x128xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%103 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%117 : tensor<?x?x32x128xf16>) outs(%118 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %108 : memref<?x?x32x128xf16>
                  %120 = loom.subview %arg3[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %121 = loom.bufferize_to_memref %119 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %122 = arith.addi %arg6, %34 : index
                  loom.copy %121, %120 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %122], LR : [%arg5, %122]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %102 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
      %c16 = arith.constant 16 : index
      %c8192 = arith.constant 8192 : index
      %21 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %22 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_n {upper_bound = 8192 : index} : index
      %24 = arith.ceildivui %c16, %21 : index
      %25 = arith.ceildivui %c8192, %22 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (4) {
            %26 = arith.ceildivui %24, %c2 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %25, %c32 : index
              scf.for %arg8 = %c0 to %27 step %c1 {
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %29 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg5, %arg6, %arg8)
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %33 = loom.subview %arg2[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %34 = arith.muli %arg4, %c4 : index
                %35 = arith.addi %34, %c3 : index
                loom.copy %33, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %34], LR : [%c7, %35]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %36 = loom.bufferize_to_tensor %32[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %37 = arith.muli %29, %22 : index
                %38 = arith.ceildivui %22, %23 : index
                %39 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %41 = loom.init_tensor %40[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %42 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %43 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %47 = loom.init_tensor %46[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %48 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %55 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %57 = loom.init_tensor %56[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %58 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %59 = loom.init_tensor %58[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %60 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %62 = loom.init_tensor %61[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %63 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %64 = loom.semaphore_take %63 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %65 = loom.init_tensor %64[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %66 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %68 = loom.init_tensor %67[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %69 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %71 = loom.init_tensor %70[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %72 = loom.alloc [%21, 128, %23] on @L1 : memref<?x128x?xf16>
                %73 = loom.semaphore_take %72 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %74 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %75 = loom.semaphore_take %74 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %76 = loom.init_tensor %75[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %77 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %78 = loom.semaphore_take %77 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %79 = loom.init_tensor %78[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %80 = loom.alloc [%21, %23, 128] on @L1 : memref<?x?x128xf16>
                %81 = loom.semaphore_take %80 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %82:3 = scf.for %arg9 = %c0 to %38 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %42) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %110 = arith.muli %arg9, %23 : index
                  %111 = arith.addi %37, %110 : index
                  %112 = loom.subview %arg0[%30, 0, %111] [%21, 128, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %113 = arith.addi %arg6, %34 : index
                  loom.copy %112, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %113], LR : [%arg5, %113]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %114 = loom.bufferize_to_tensor %73[%21, 128, %23] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%76 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %116 = linalg.batch_matmul ins(%36, %114 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%115 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %73 : memref<?x128x?xf16>
                  %117 = linalg.fill ins(%cst_1 : f16) outs(%65 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%65 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in_3, %cst_2 : f16
                    %134 = arith.cmpf ogt, %in, %133 : f16
                    %135 = arith.select %134, %in, %133 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%79 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %120 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%76 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in, %cst_2 : f16
                    %134 = arith.subf %133, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %78 : memref<?x32x?xf16>
                  %122 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121 : tensor<?x32x?xf16>) outs(%122 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%71 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.subf %in, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %124, %123 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in, %in_3 : f16
                    %134 = arith.addf %133, %in_4 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %67 : memref<?x32x1xf16>
                  %126 = loom.subview %arg1[%30, %111, 0] [%21, %23, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %126, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %113], LR : [%arg5, %113]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %127 = loom.bufferize_to_tensor %81[%21, %23, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %129 = linalg.batch_matmul ins(%121, %127 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%128 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %81 : memref<?x?x128xf16>
                  loom.semaphore_give %75 : memref<?x32x?xf16>
                  %130 = loom.broadcast ins(%124 : tensor<?x32x1xf16>) outs(%62 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %70 : memref<?x32x1xf16>
                  %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %arg12, %130 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in_3, %in_4 : f16
                    %134 = arith.addf %in, %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %61 : memref<?x32x128xf16>
                  loom.semaphore_give %58 : memref<?x32x128xf16>
                  %132 = linalg.copy ins(%119 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %64 : memref<?x32x1xf16>
                  scf.yield %132, %125, %131 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %32 : memref<?x32x128xf16>
                %83 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %84 = loom.semaphore_take %83 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %85 = loom.init_tensor %84[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82#1, %82#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%85 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = math.log %in : f16
                  %111 = arith.addf %110, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %52 : memref<?x32x1xf16>
                %87 = loom.broadcast ins(%82#1 : tensor<?x32x1xf16>) outs(%57 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %88 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %89 = loom.semaphore_take %88 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %90 = loom.init_tensor %89[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82#2, %87 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%90 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = arith.divf %in, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %56 : memref<?x32x128xf16>
                loom.semaphore_give %40 : memref<?x32x128xf16>
                %92 = loom.bufferize_to_memref %86 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %93 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %92, %94 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [8, 4] region : (UL : [%c0, %34], LR : [%c7, %35]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %84 : memref<?x32x1xf16>
                %95 = loom.bufferize_to_tensor %94[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %96 = loom.bufferize_to_memref %91 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %97 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %98 = loom.semaphore_take %97 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %96, %98 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [8, 4] region : (UL : [%c0, %34], LR : [%c7, %35]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %89 : memref<?x32x128xf16>
                %99 = loom.bufferize_to_tensor %98[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %100 = arith.cmpi eq, %29, %c0 : index
                %101 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %102 = loom.semaphore_take %101 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %103 = loom.init_tensor %102[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %104 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %106 = loom.init_tensor %105[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %107 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %108 = loom.semaphore_take %107 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %109 = loom.init_tensor %108[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                scf.if %100 {
                  %110 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%95 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.maximumf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %111 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.subf %in, %in_3 : f16
                    %124 = math.exp %123 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %94 : memref<?x?x32x1xf16>
                  loom.semaphore_give %46 : memref<?x32x1xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%112 : tensor<?x?x32x1xf16>) outs(%113 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %114 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.divf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %44 : memref<?x32x1xf16>
                  %116 = loom.broadcast ins(%115 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %116 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%109 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.mulf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %98 : memref<?x?x32x128xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%103 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%117 : tensor<?x?x32x128xf16>) outs(%118 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %108 : memref<?x?x32x128xf16>
                  %120 = loom.subview %arg3[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %121 = loom.bufferize_to_memref %119 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %122 = arith.addi %arg6, %34 : index
                  loom.copy %121, %120 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %122], LR : [%arg5, %122]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %102 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
      %c16 = arith.constant 16 : index
      %c8192 = arith.constant 8192 : index
      %21 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %22 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_n {upper_bound = 8192 : index} : index
      %24 = arith.ceildivui %c16, %21 : index
      %25 = arith.ceildivui %c8192, %22 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %24 step %c1 {
              %26 = arith.ceildivui %25, %c64 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %28 = arith.muli %arg7, %21 : index
                %29 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %31 = loom.subview %arg2[%28, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %32 = loom.bufferize_to_tensor %30[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %33 = arith.muli %27, %22 : index
                %34 = arith.ceildivui %22, %23 : index
                %35 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %36 = loom.semaphore_take %35 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %37 = loom.init_tensor %36[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %39 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %41 = loom.init_tensor %40[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %42 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %43 = loom.init_tensor %42[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %44 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = linalg.fill ins(%cst_0 : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %47 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %53 = loom.init_tensor %52[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %54 = loom.semaphore_take %51 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %55 = loom.init_tensor %54[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %56 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %57 = loom.semaphore_take %56 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %58 = loom.init_tensor %57[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %59 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %61 = loom.init_tensor %60[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %62 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %63 = loom.semaphore_take %62 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %64 = loom.init_tensor %63[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %65 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %67 = loom.init_tensor %66[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %68 = loom.alloc [%21, 128, %23] on @L1 : memref<?x128x?xf16>
                %69 = loom.semaphore_take %68 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %70 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %71 = loom.semaphore_take %70 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %72 = loom.init_tensor %71[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %73 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %74 = loom.semaphore_take %73 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %75 = loom.init_tensor %74[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %76 = loom.alloc [%21, %23, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %34 step %c1 iter_args(%arg10 = %50, %arg11 = %46, %arg12 = %38) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %108 = arith.muli %arg9, %23 : index
                  %109 = arith.addi %33, %108 : index
                  %110 = loom.subview %arg0[%28, 0, %109] [%21, 128, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %111 = arith.muli %arg4, %c8 : index
                  %112 = arith.addi %arg6, %111 : index
                  loom.copy %110, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %112], LR : [%arg5, %112]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %113 = loom.bufferize_to_tensor %69[%21, 128, %23] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%72 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %115 = linalg.batch_matmul ins(%32, %113 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%114 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %69 : memref<?x128x?xf16>
                  %116 = linalg.fill ins(%cst_1 : f16) outs(%61 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%115 : tensor<?x32x?xf16>) outs(%116 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.maximumf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%61 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in_3, %cst_2 : f16
                    %133 = arith.cmpf ogt, %in, %132 : f16
                    %134 = arith.select %133, %in, %132 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = loom.broadcast ins(%118 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %119 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%72 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in, %cst_2 : f16
                    %133 = arith.subf %132, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x?xf16>
                  %121 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%120 : tensor<?x32x?xf16>) outs(%121 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.addf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%67 : tensor<?x32x1xf16>) {
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
                  loom.semaphore_give %63 : memref<?x32x1xf16>
                  %125 = loom.subview %arg1[%28, %109, 0] [%21, %23, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %125, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %112], LR : [%arg5, %112]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %126 = loom.bufferize_to_tensor %77[%21, %23, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %127 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %128 = linalg.batch_matmul ins(%120, %126 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%127 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %71 : memref<?x32x?xf16>
                  %129 = loom.broadcast ins(%123 : tensor<?x32x1xf16>) outs(%58 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %66 : memref<?x32x1xf16>
                  %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %arg12, %129 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in_3, %in_4 : f16
                    %133 = arith.addf %in, %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %57 : memref<?x32x128xf16>
                  loom.semaphore_give %54 : memref<?x32x128xf16>
                  %131 = linalg.copy ins(%118 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %60 : memref<?x32x1xf16>
                  scf.yield %131, %124, %130 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %30 : memref<?x32x128xf16>
                %79 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %108 = math.log %in : f16
                  %109 = arith.addf %108, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%53 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %44 : memref<?x32x1xf16>
                %84 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %108 = arith.divf %in, %in_3 : f16
                  linalg.yield %108 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %52 : memref<?x32x128xf16>
                loom.semaphore_give %36 : memref<?x32x128xf16>
                %88 = loom.bufferize_to_memref %82 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %89 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %90 = loom.semaphore_take %89 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %91 = arith.muli %arg4, %c8 : index
                %92 = arith.addi %91, %c7 : index
                loom.gather %88, %90 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%27 : index), area : [8, 8] region : (UL : [%c0, %91], LR : [%c7, %92]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %93 = loom.bufferize_to_tensor %90[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %94 = loom.bufferize_to_memref %87 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %95 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %96 = loom.semaphore_take %95 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %94, %96 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%27 : index), area : [8, 8] region : (UL : [%c0, %91], LR : [%c7, %92]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %97 = loom.bufferize_to_tensor %96[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %98 = arith.cmpi eq, %27, %c0 : index
                %99 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %100 = loom.semaphore_take %99 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %101 = loom.init_tensor %100[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %102 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %104 = loom.init_tensor %103[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %105 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %107 = loom.init_tensor %106[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                scf.if %98 {
                  %108 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%93 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.maximumf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %109 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.subf %in, %in_3 : f16
                    %122 = math.exp %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %90 : memref<?x?x32x1xf16>
                  loom.semaphore_give %42 : memref<?x32x1xf16>
                  %111 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%110 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %112 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.divf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %40 : memref<?x32x1xf16>
                  %114 = loom.broadcast ins(%113 : tensor<?x?x32x1xf16>) outs(%107 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %114 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%107 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.mulf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %96 : memref<?x?x32x128xf16>
                  %116 = linalg.fill ins(%cst : f16) outs(%101 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%115 : tensor<?x?x32x128xf16>) outs(%116 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x128xf16>
                  %118 = loom.subview %arg3[%28, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %119 = loom.bufferize_to_memref %117 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %120 = arith.addi %arg6, %91 : index
                  loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %120], LR : [%arg5, %120]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %100 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x1x8_y8__d0i1_d1i1_d2i0__f01__dim_y_level0_bc8_n_n_n(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c16 = arith.constant 16 : index
      %c8192 = arith.constant 8192 : index
      %21 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %22 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_n {upper_bound = 8192 : index} : index
      %24 = arith.ceildivui %c16, %21 : index
      %25 = arith.ceildivui %c8192, %22 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (1) {
          affine.parallel (%arg6) = (0) to (8) {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %25, %c8 : index
              scf.for %arg8 = %c0 to %27 step %c1 {
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %29 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg6, %arg8)
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %33 = loom.subview %arg2[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %34 = arith.addi %arg5, %arg4 : index
                loom.copy %33, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%34, %c0], LR : [%34, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %35 = loom.bufferize_to_tensor %32[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %36 = arith.muli %29, %22 : index
                %37 = arith.ceildivui %22, %23 : index
                %38 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %39 = loom.semaphore_take %38 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %40 = loom.init_tensor %39[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %41 = linalg.fill ins(%cst : f16) outs(%40 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %42 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %44 = loom.init_tensor %43[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %45 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %46 = loom.init_tensor %45[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %47 = loom.semaphore_take %42 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %48 = loom.init_tensor %47[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %49 = linalg.fill ins(%cst_0 : f16) outs(%48 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %50 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %51 = loom.semaphore_take %50 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %52 = loom.init_tensor %51[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %53 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %54 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %55 = loom.semaphore_take %54 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %56 = loom.init_tensor %55[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %57 = loom.semaphore_take %54 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %58 = loom.init_tensor %57[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %59 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %60 = loom.semaphore_take %59 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %61 = loom.init_tensor %60[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %62 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %63 = loom.semaphore_take %62 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %64 = loom.init_tensor %63[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %65 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %67 = loom.init_tensor %66[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %68 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %69 = loom.semaphore_take %68 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %70 = loom.init_tensor %69[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %71 = loom.alloc [%21, 128, %23] on @L1 : memref<?x128x?xf16>
                %72 = loom.semaphore_take %71 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %73 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %74 = loom.semaphore_take %73 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %75 = loom.init_tensor %74[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %76 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %77 = loom.semaphore_take %76 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %78 = loom.init_tensor %77[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %79 = loom.alloc [%21, %23, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81:3 = scf.for %arg9 = %c0 to %37 step %c1 iter_args(%arg10 = %53, %arg11 = %49, %arg12 = %41) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %109 = arith.muli %arg9, %23 : index
                  %110 = arith.addi %36, %109 : index
                  %111 = loom.subview %arg0[%30, 0, %110] [%21, 128, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  loom.copy %111, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%34, %arg6], LR : [%34, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %112 = loom.bufferize_to_tensor %72[%21, 128, %23] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%75 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %114 = linalg.batch_matmul ins(%35, %112 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%113 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %72 : memref<?x128x?xf16>
                  %115 = linalg.fill ins(%cst_1 : f16) outs(%64 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%114 : tensor<?x32x?xf16>) outs(%115 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.maximumf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %116 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%64 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in_3, %cst_2 : f16
                    %132 = arith.cmpf ogt, %in, %131 : f16
                    %133 = arith.select %132, %in, %131 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = loom.broadcast ins(%117 : tensor<?x32x1xf16>) outs(%78 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%114, %118 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%75 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %131 = arith.mulf %in, %cst_2 : f16
                    %132 = arith.subf %131, %in_3 : f16
                    %133 = math.exp %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %77 : memref<?x32x?xf16>
                  %120 = linalg.fill ins(%cst : f16) outs(%67 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%119 : tensor<?x32x?xf16>) outs(%120 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %131 = arith.addf %in, %out : f16
                    linalg.yield %131 : f16
                  } -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%70 : tensor<?x32x1xf16>) {
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
                  loom.semaphore_give %66 : memref<?x32x1xf16>
                  %124 = loom.subview %arg1[%30, %110, 0] [%21, %23, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %124, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%34, %arg6], LR : [%34, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %125 = loom.bufferize_to_tensor %80[%21, %23, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %126 = linalg.fill ins(%cst : f16) outs(%58 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %127 = linalg.batch_matmul ins(%119, %125 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%126 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %74 : memref<?x32x?xf16>
                  %128 = loom.broadcast ins(%122 : tensor<?x32x1xf16>) outs(%61 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %69 : memref<?x32x1xf16>
                  %129 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%127, %arg12, %128 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %131 = arith.mulf %in_3, %in_4 : f16
                    %132 = arith.addf %in, %131 : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %60 : memref<?x32x128xf16>
                  loom.semaphore_give %57 : memref<?x32x128xf16>
                  %130 = linalg.copy ins(%117 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %63 : memref<?x32x1xf16>
                  scf.yield %130, %123, %129 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %32 : memref<?x32x128xf16>
                %82 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %83 = loom.semaphore_take %82 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %84 = loom.init_tensor %83[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#1, %81#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%84 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = math.log %in : f16
                  %110 = arith.addf %109, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %51 : memref<?x32x1xf16>
                %86 = loom.broadcast ins(%81#1 : tensor<?x32x1xf16>) outs(%56 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %47 : memref<?x32x1xf16>
                %87 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %88 = loom.semaphore_take %87 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %89 = loom.init_tensor %88[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#2, %86 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%89 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %109 = arith.divf %in, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %55 : memref<?x32x128xf16>
                loom.semaphore_give %39 : memref<?x32x128xf16>
                %91 = loom.bufferize_to_memref %85 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %92 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %93 = loom.semaphore_take %92 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %91, %93 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %83 : memref<?x32x1xf16>
                %94 = loom.bufferize_to_tensor %93[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %95 = loom.bufferize_to_memref %90 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %96 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %97 = loom.semaphore_take %96 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %95, %97 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %88 : memref<?x32x128xf16>
                %98 = loom.bufferize_to_tensor %97[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %99 = arith.cmpi eq, %29, %c0 : index
                %100 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %101 = loom.semaphore_take %100 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %102 = loom.init_tensor %101[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %103 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %104 = loom.semaphore_take %103 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %105 = loom.init_tensor %104[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %106 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %107 = loom.semaphore_take %106 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %108 = loom.init_tensor %107[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                scf.if %99 {
                  %109 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%94 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.maximumf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %110 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%105 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.subf %in, %in_3 : f16
                    %122 = math.exp %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %93 : memref<?x?x32x1xf16>
                  loom.semaphore_give %45 : memref<?x32x1xf16>
                  %112 = linalg.fill ins(%cst : f16) outs(%44 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%111 : tensor<?x?x32x1xf16>) outs(%112 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%111, %113 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%105 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.divf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %43 : memref<?x32x1xf16>
                  %115 = loom.broadcast ins(%114 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %104 : memref<?x?x32x1xf16>
                  %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %115 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%108 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.mulf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %97 : memref<?x?x32x128xf16>
                  %117 = linalg.fill ins(%cst : f16) outs(%102 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%116 : tensor<?x?x32x128xf16>) outs(%117 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %107 : memref<?x?x32x128xf16>
                  %119 = loom.subview %arg3[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %120 = loom.bufferize_to_memref %118 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  loom.copy %120, %119 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%34, %arg6], LR : [%34, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %101 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
      %c16 = arith.constant 16 : index
      %c8192 = arith.constant 8192 : index
      %21 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %22 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_n {upper_bound = 8192 : index} : index
      %24 = arith.ceildivui %c16, %21 : index
      %25 = arith.ceildivui %c8192, %22 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (2) {
          affine.parallel (%arg6) = (0) to (8) {
            %26 = arith.ceildivui %24, %c4 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %25, %c16 : index
              scf.for %arg8 = %c0 to %27 step %c1 {
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %29 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg5, %arg6, %arg8)
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %33 = loom.subview %arg2[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %34 = arith.muli %arg4, %c2 : index
                %35 = arith.addi %34, %c1 : index
                loom.copy %33, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%34, %c0], LR : [%35, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %36 = loom.bufferize_to_tensor %32[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %37 = arith.muli %29, %22 : index
                %38 = arith.ceildivui %22, %23 : index
                %39 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %41 = loom.init_tensor %40[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %42 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %43 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %47 = loom.init_tensor %46[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %48 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %55 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %57 = loom.init_tensor %56[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %58 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %59 = loom.init_tensor %58[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %60 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %62 = loom.init_tensor %61[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %63 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %64 = loom.semaphore_take %63 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %65 = loom.init_tensor %64[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %66 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %68 = loom.init_tensor %67[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %69 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %71 = loom.init_tensor %70[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %72 = loom.alloc [%21, 128, %23] on @L1 : memref<?x128x?xf16>
                %73 = loom.semaphore_take %72 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %74 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %75 = loom.semaphore_take %74 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %76 = loom.init_tensor %75[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %77 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %78 = loom.semaphore_take %77 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %79 = loom.init_tensor %78[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %80 = loom.alloc [%21, %23, 128] on @L1 : memref<?x?x128xf16>
                %81 = loom.semaphore_take %80 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %82:3 = scf.for %arg9 = %c0 to %38 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %42) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %110 = arith.muli %arg9, %23 : index
                  %111 = arith.addi %37, %110 : index
                  %112 = loom.subview %arg0[%30, 0, %111] [%21, 128, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %113 = arith.addi %arg5, %34 : index
                  loom.copy %112, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%113, %arg6], LR : [%113, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %114 = loom.bufferize_to_tensor %73[%21, 128, %23] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%76 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %116 = linalg.batch_matmul ins(%36, %114 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%115 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %73 : memref<?x128x?xf16>
                  %117 = linalg.fill ins(%cst_1 : f16) outs(%65 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%65 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in_3, %cst_2 : f16
                    %134 = arith.cmpf ogt, %in, %133 : f16
                    %135 = arith.select %134, %in, %133 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%79 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %120 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%76 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in, %cst_2 : f16
                    %134 = arith.subf %133, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %78 : memref<?x32x?xf16>
                  %122 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121 : tensor<?x32x?xf16>) outs(%122 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%71 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.subf %in, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %124, %123 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in, %in_3 : f16
                    %134 = arith.addf %133, %in_4 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %67 : memref<?x32x1xf16>
                  %126 = loom.subview %arg1[%30, %111, 0] [%21, %23, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %126, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%113, %arg6], LR : [%113, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %127 = loom.bufferize_to_tensor %81[%21, %23, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %129 = linalg.batch_matmul ins(%121, %127 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%128 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %81 : memref<?x?x128xf16>
                  loom.semaphore_give %75 : memref<?x32x?xf16>
                  %130 = loom.broadcast ins(%124 : tensor<?x32x1xf16>) outs(%62 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %70 : memref<?x32x1xf16>
                  %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %arg12, %130 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in_3, %in_4 : f16
                    %134 = arith.addf %in, %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %61 : memref<?x32x128xf16>
                  loom.semaphore_give %58 : memref<?x32x128xf16>
                  %132 = linalg.copy ins(%119 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %64 : memref<?x32x1xf16>
                  scf.yield %132, %125, %131 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %32 : memref<?x32x128xf16>
                %83 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %84 = loom.semaphore_take %83 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %85 = loom.init_tensor %84[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82#1, %82#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%85 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = math.log %in : f16
                  %111 = arith.addf %110, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %52 : memref<?x32x1xf16>
                %87 = loom.broadcast ins(%82#1 : tensor<?x32x1xf16>) outs(%57 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %88 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %89 = loom.semaphore_take %88 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %90 = loom.init_tensor %89[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82#2, %87 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%90 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = arith.divf %in, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %56 : memref<?x32x128xf16>
                loom.semaphore_give %40 : memref<?x32x128xf16>
                %92 = loom.bufferize_to_memref %86 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %93 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %92, %94 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [2, 8] region : (UL : [%34, %c0], LR : [%35, %c7]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %84 : memref<?x32x1xf16>
                %95 = loom.bufferize_to_tensor %94[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %96 = loom.bufferize_to_memref %91 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %97 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %98 = loom.semaphore_take %97 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %96, %98 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [2, 8] region : (UL : [%34, %c0], LR : [%35, %c7]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %89 : memref<?x32x128xf16>
                %99 = loom.bufferize_to_tensor %98[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %100 = arith.cmpi eq, %29, %c0 : index
                %101 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %102 = loom.semaphore_take %101 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %103 = loom.init_tensor %102[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %104 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %106 = loom.init_tensor %105[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %107 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %108 = loom.semaphore_take %107 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %109 = loom.init_tensor %108[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                scf.if %100 {
                  %110 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%95 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.maximumf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %111 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.subf %in, %in_3 : f16
                    %124 = math.exp %123 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %94 : memref<?x?x32x1xf16>
                  loom.semaphore_give %46 : memref<?x32x1xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%112 : tensor<?x?x32x1xf16>) outs(%113 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %114 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.divf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %44 : memref<?x32x1xf16>
                  %116 = loom.broadcast ins(%115 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %116 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%109 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.mulf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %98 : memref<?x?x32x128xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%103 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%117 : tensor<?x?x32x128xf16>) outs(%118 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %108 : memref<?x?x32x128xf16>
                  %120 = loom.subview %arg3[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %121 = loom.bufferize_to_memref %119 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %122 = arith.addi %arg5, %34 : index
                  loom.copy %121, %120 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%122, %arg6], LR : [%122, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %102 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
      %c16 = arith.constant 16 : index
      %c8192 = arith.constant 8192 : index
      %21 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %22 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_n {upper_bound = 8192 : index} : index
      %24 = arith.ceildivui %c16, %21 : index
      %25 = arith.ceildivui %c8192, %22 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (4) {
          affine.parallel (%arg6) = (0) to (8) {
            %26 = arith.ceildivui %24, %c2 : index
            scf.for %arg7 = %c0 to %26 step %c1 {
              %27 = arith.ceildivui %25, %c32 : index
              scf.for %arg8 = %c0 to %27 step %c1 {
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %29 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg5, %arg6, %arg8)
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %33 = loom.subview %arg2[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                %34 = arith.muli %arg4, %c4 : index
                %35 = arith.addi %34, %c3 : index
                loom.copy %33, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%34, %c0], LR : [%35, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %36 = loom.bufferize_to_tensor %32[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %37 = arith.muli %29, %22 : index
                %38 = arith.ceildivui %22, %23 : index
                %39 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %41 = loom.init_tensor %40[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %42 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %43 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %47 = loom.init_tensor %46[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %48 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %53 = loom.init_tensor %52[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %55 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %57 = loom.init_tensor %56[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %58 = loom.semaphore_take %55 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %59 = loom.init_tensor %58[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %60 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %61 = loom.semaphore_take %60 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %62 = loom.init_tensor %61[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %63 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %64 = loom.semaphore_take %63 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %65 = loom.init_tensor %64[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %66 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %67 = loom.semaphore_take %66 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %68 = loom.init_tensor %67[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %69 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %70 = loom.semaphore_take %69 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %71 = loom.init_tensor %70[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %72 = loom.alloc [%21, 128, %23] on @L1 : memref<?x128x?xf16>
                %73 = loom.semaphore_take %72 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %74 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %75 = loom.semaphore_take %74 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %76 = loom.init_tensor %75[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %77 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %78 = loom.semaphore_take %77 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %79 = loom.init_tensor %78[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %80 = loom.alloc [%21, %23, 128] on @L1 : memref<?x?x128xf16>
                %81 = loom.semaphore_take %80 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %82:3 = scf.for %arg9 = %c0 to %38 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %42) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %110 = arith.muli %arg9, %23 : index
                  %111 = arith.addi %37, %110 : index
                  %112 = loom.subview %arg0[%30, 0, %111] [%21, 128, %23] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %113 = arith.addi %arg5, %34 : index
                  loom.copy %112, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%113, %arg6], LR : [%113, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %114 = loom.bufferize_to_tensor %73[%21, 128, %23] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %115 = linalg.fill ins(%cst : f16) outs(%76 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %116 = linalg.batch_matmul ins(%36, %114 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%115 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %73 : memref<?x128x?xf16>
                  %117 = linalg.fill ins(%cst_1 : f16) outs(%65 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%116 : tensor<?x32x?xf16>) outs(%117 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.maximumf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%65 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in_3, %cst_2 : f16
                    %134 = arith.cmpf ogt, %in, %133 : f16
                    %135 = arith.select %134, %in, %133 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x1xf16>
                  %120 = loom.broadcast ins(%119 : tensor<?x32x1xf16>) outs(%79 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
                  %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%116, %120 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%76 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.mulf %in, %cst_2 : f16
                    %134 = arith.subf %133, %in_3 : f16
                    %135 = math.exp %134 : f16
                    linalg.yield %135 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %78 : memref<?x32x?xf16>
                  %122 = linalg.fill ins(%cst : f16) outs(%68 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%121 : tensor<?x32x?xf16>) outs(%122 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %133 = arith.addf %in, %out : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x1xf16>
                  %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %119 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%71 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %133 = arith.subf %in, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %125 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %124, %123 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg11 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in, %in_3 : f16
                    %134 = arith.addf %133, %in_4 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  loom.semaphore_give %67 : memref<?x32x1xf16>
                  %126 = loom.subview %arg1[%30, %111, 0] [%21, %23, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %126, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%113, %arg6], LR : [%113, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %127 = loom.bufferize_to_tensor %81[%21, %23, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %128 = linalg.fill ins(%cst : f16) outs(%59 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %129 = linalg.batch_matmul ins(%121, %127 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%128 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %81 : memref<?x?x128xf16>
                  loom.semaphore_give %75 : memref<?x32x?xf16>
                  %130 = loom.broadcast ins(%124 : tensor<?x32x1xf16>) outs(%62 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %70 : memref<?x32x1xf16>
                  %131 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%129, %arg12, %130 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %133 = arith.mulf %in_3, %in_4 : f16
                    %134 = arith.addf %in, %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %61 : memref<?x32x128xf16>
                  loom.semaphore_give %58 : memref<?x32x128xf16>
                  %132 = linalg.copy ins(%119 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %64 : memref<?x32x1xf16>
                  scf.yield %132, %125, %131 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %32 : memref<?x32x128xf16>
                %83 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %84 = loom.semaphore_take %83 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %85 = loom.init_tensor %84[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82#1, %82#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%85 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = math.log %in : f16
                  %111 = arith.addf %110, %in_3 : f16
                  linalg.yield %111 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %52 : memref<?x32x1xf16>
                %87 = loom.broadcast ins(%82#1 : tensor<?x32x1xf16>) outs(%57 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %88 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %89 = loom.semaphore_take %88 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %90 = loom.init_tensor %89[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82#2, %87 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%90 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %110 = arith.divf %in, %in_3 : f16
                  linalg.yield %110 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %56 : memref<?x32x128xf16>
                loom.semaphore_give %40 : memref<?x32x128xf16>
                %92 = loom.bufferize_to_memref %86 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %93 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %94 = loom.semaphore_take %93 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                loom.gather %92, %94 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [4, 8] region : (UL : [%34, %c0], LR : [%35, %c7]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %84 : memref<?x32x1xf16>
                %95 = loom.bufferize_to_tensor %94[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %96 = loom.bufferize_to_memref %91 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %97 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %98 = loom.semaphore_take %97 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %96, %98 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%29 : index), area : [4, 8] region : (UL : [%34, %c0], LR : [%35, %c7]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %89 : memref<?x32x128xf16>
                %99 = loom.bufferize_to_tensor %98[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %100 = arith.cmpi eq, %29, %c0 : index
                %101 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %102 = loom.semaphore_take %101 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %103 = loom.init_tensor %102[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %104 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %105 = loom.semaphore_take %104 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %106 = loom.init_tensor %105[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %107 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %108 = loom.semaphore_take %107 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %109 = loom.init_tensor %108[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                scf.if %100 {
                  %110 = linalg.fill ins(%cst_1 : f16) outs(%47 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%95 : tensor<?x?x32x1xf16>) outs(%110 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.maximumf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %111 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.subf %in, %in_3 : f16
                    %124 = math.exp %123 : f16
                    linalg.yield %124 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %94 : memref<?x?x32x1xf16>
                  loom.semaphore_give %46 : memref<?x32x1xf16>
                  %113 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%112 : tensor<?x?x32x1xf16>) outs(%113 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %114 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%106 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.divf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %44 : memref<?x32x1xf16>
                  %116 = loom.broadcast ins(%115 : tensor<?x?x32x1xf16>) outs(%109 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %105 : memref<?x?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %116 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%109 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %123 = arith.mulf %in, %in_3 : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %98 : memref<?x?x32x128xf16>
                  %118 = linalg.fill ins(%cst : f16) outs(%103 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %119 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%117 : tensor<?x?x32x128xf16>) outs(%118 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %123 = arith.addf %in, %out : f16
                    linalg.yield %123 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %108 : memref<?x?x32x128xf16>
                  %120 = loom.subview %arg3[%30, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %121 = loom.bufferize_to_memref %119 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %122 = arith.addi %arg5, %34 : index
                  loom.copy %121, %120 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%122, %arg6], LR : [%122, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %102 : memref<?x32x128xf16>
                }
              } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
      %c16 = arith.constant 16 : index
      %c8192 = arith.constant 8192 : index
      %21 = loom.sym @tile_b {upper_bound = 16 : index} : index
      %22 = loom.sym @tile_s {upper_bound = 8192 : index} : index
      %23 = loom.sym @tile_n {upper_bound = 8192 : index} : index
      %24 = arith.ceildivui %c16, %21 : index
      %25 = arith.ceildivui %c8192, %22 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %24 step %c1 {
              %26 = arith.ceildivui %25, %c64 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %28 = arith.muli %arg7, %21 : index
                %29 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %31 = loom.subview %arg2[%28, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
                %32 = loom.bufferize_to_tensor %30[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %33 = arith.muli %27, %22 : index
                %34 = arith.ceildivui %22, %23 : index
                %35 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %36 = loom.semaphore_take %35 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %37 = loom.init_tensor %36[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                %39 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %40 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %41 = loom.init_tensor %40[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %42 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %43 = loom.init_tensor %42[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %44 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %45 = loom.init_tensor %44[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %46 = linalg.fill ins(%cst_0 : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %47 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %49 = loom.init_tensor %48[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %50 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                %51 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %52 = loom.semaphore_take %51 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %53 = loom.init_tensor %52[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %54 = loom.semaphore_take %51 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %55 = loom.init_tensor %54[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %56 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %57 = loom.semaphore_take %56 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %58 = loom.init_tensor %57[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %59 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %61 = loom.init_tensor %60[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %62 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %63 = loom.semaphore_take %62 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %64 = loom.init_tensor %63[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %65 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %66 = loom.semaphore_take %65 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %67 = loom.init_tensor %66[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %68 = loom.alloc [%21, 128, %23] on @L1 : memref<?x128x?xf16>
                %69 = loom.semaphore_take %68 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %70 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %71 = loom.semaphore_take %70 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %72 = loom.init_tensor %71[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %73 = loom.alloc [%21, 32, %23] on @L1 : memref<?x32x?xf16>
                %74 = loom.semaphore_take %73 : memref<?x32x?xf16> -> memref<?x32x?xf16>
                %75 = loom.init_tensor %74[%21, 32, %23] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
                %76 = loom.alloc [%21, %23, 128] on @L1 : memref<?x?x128xf16>
                %77 = loom.semaphore_take %76 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %78:3 = scf.for %arg9 = %c0 to %34 step %c1 iter_args(%arg10 = %50, %arg11 = %46, %arg12 = %38) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
                  %108 = arith.muli %arg9, %23 : index
                  %109 = arith.addi %33, %108 : index
                  %110 = loom.subview %arg0[%28, 0, %109] [%21, 128, %23] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
                  %111 = arith.muli %arg4, %c8 : index
                  %112 = arith.addi %arg5, %111 : index
                  loom.copy %110, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%112, %arg6], LR : [%112, %arg6]) : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
                  %113 = loom.bufferize_to_tensor %69[%21, 128, %23] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %114 = linalg.fill ins(%cst : f16) outs(%72 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  %115 = linalg.batch_matmul ins(%32, %113 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%114 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
                  loom.semaphore_give %69 : memref<?x128x?xf16>
                  %116 = linalg.fill ins(%cst_1 : f16) outs(%61 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%115 : tensor<?x32x?xf16>) outs(%116 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.maximumf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %117 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%61 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in_3, %cst_2 : f16
                    %133 = arith.cmpf ogt, %in, %132 : f16
                    %134 = arith.select %133, %in, %132 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x1xf16>
                  %119 = loom.broadcast ins(%118 : tensor<?x32x1xf16>) outs(%75 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
                  %120 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %119 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%72 : tensor<?x32x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %132 = arith.mulf %in, %cst_2 : f16
                    %133 = arith.subf %132, %in_3 : f16
                    %134 = math.exp %133 : f16
                    linalg.yield %134 : f16
                  } -> tensor<?x32x?xf16>
                  loom.semaphore_give %74 : memref<?x32x?xf16>
                  %121 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %122 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%120 : tensor<?x32x?xf16>) outs(%121 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %132 = arith.addf %in, %out : f16
                    linalg.yield %132 : f16
                  } -> tensor<?x32x1xf16>
                  %123 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %118 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%67 : tensor<?x32x1xf16>) {
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
                  loom.semaphore_give %63 : memref<?x32x1xf16>
                  %125 = loom.subview %arg1[%28, %109, 0] [%21, %23, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
                  loom.copy %125, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%112, %arg6], LR : [%112, %arg6]) : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %126 = loom.bufferize_to_tensor %77[%21, %23, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %127 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %128 = linalg.batch_matmul ins(%120, %126 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%127 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  loom.semaphore_give %77 : memref<?x?x128xf16>
                  loom.semaphore_give %71 : memref<?x32x?xf16>
                  %129 = loom.broadcast ins(%123 : tensor<?x32x1xf16>) outs(%58 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                  loom.semaphore_give %66 : memref<?x32x1xf16>
                  %130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%128, %arg12, %129 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg12 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %132 = arith.mulf %in_3, %in_4 : f16
                    %133 = arith.addf %in, %132 : f16
                    linalg.yield %133 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %57 : memref<?x32x128xf16>
                  loom.semaphore_give %54 : memref<?x32x128xf16>
                  %131 = linalg.copy ins(%118 : tensor<?x32x1xf16>) outs(%arg10 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  loom.semaphore_give %60 : memref<?x32x1xf16>
                  scf.yield %131, %124, %130 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %30 : memref<?x32x128xf16>
                %79 = loom.alloc [%21, 32, 1] on @L1 : memref<?x32x1xf16>
                %80 = loom.semaphore_take %79 : memref<?x32x1xf16> -> memref<?x32x1xf16>
                %81 = loom.init_tensor %80[%21, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#1, %78#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x32x1xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %108 = math.log %in : f16
                  %109 = arith.addf %108, %in_3 : f16
                  linalg.yield %109 : f16
                } -> tensor<?x32x1xf16>
                loom.semaphore_give %48 : memref<?x32x1xf16>
                %83 = loom.broadcast ins(%78#1 : tensor<?x32x1xf16>) outs(%53 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
                loom.semaphore_give %44 : memref<?x32x1xf16>
                %84 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %85 = loom.semaphore_take %84 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %86 = loom.init_tensor %85[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78#2, %83 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%86 : tensor<?x32x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %108 = arith.divf %in, %in_3 : f16
                  linalg.yield %108 : f16
                } -> tensor<?x32x128xf16>
                loom.semaphore_give %52 : memref<?x32x128xf16>
                loom.semaphore_give %36 : memref<?x32x128xf16>
                %88 = loom.bufferize_to_memref %82 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
                %89 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %90 = loom.semaphore_take %89 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %91 = arith.muli %arg4, %c8 : index
                %92 = arith.addi %91, %c7 : index
                loom.gather %88, %90 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%27 : index), area : [8, 8] region : (UL : [%91, %c0], LR : [%92, %c7]) : memref<?x32x1xf16> to memref<?x?x32x1xf16>
                loom.semaphore_give %80 : memref<?x32x1xf16>
                %93 = loom.bufferize_to_tensor %90[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %94 = loom.bufferize_to_memref %87 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                %95 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %96 = loom.semaphore_take %95 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                loom.gather %94, %96 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%27 : index), area : [8, 8] region : (UL : [%91, %c0], LR : [%92, %c7]) : memref<?x32x128xf16> to memref<?x?x32x128xf16>
                loom.semaphore_give %85 : memref<?x32x128xf16>
                %97 = loom.bufferize_to_tensor %96[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                %98 = arith.cmpi eq, %27, %c0 : index
                %99 = loom.alloc [%21, 32, 128] on @L1 : memref<?x32x128xf16>
                %100 = loom.semaphore_take %99 : memref<?x32x128xf16> -> memref<?x32x128xf16>
                %101 = loom.init_tensor %100[%21, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
                %102 = loom.alloc [%25, %21, 32, 1] on @L1 : memref<?x?x32x1xf16>
                %103 = loom.semaphore_take %102 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
                %104 = loom.init_tensor %103[%25, %21, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
                %105 = loom.alloc [%25, %21, 32, 128] on @L1 : memref<?x?x32x128xf16>
                %106 = loom.semaphore_take %105 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
                %107 = loom.init_tensor %106[%25, %21, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
                scf.if %98 {
                  %108 = linalg.fill ins(%cst_1 : f16) outs(%43 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%93 : tensor<?x?x32x1xf16>) outs(%108 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.maximumf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %109 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.subf %in, %in_3 : f16
                    %122 = math.exp %121 : f16
                    linalg.yield %122 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %90 : memref<?x?x32x1xf16>
                  loom.semaphore_give %42 : memref<?x32x1xf16>
                  %111 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
                  %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%110 : tensor<?x?x32x1xf16>) outs(%111 : tensor<?x32x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x1xf16>
                  %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%110, %112 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%104 : tensor<?x?x32x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.divf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x1xf16>
                  loom.semaphore_give %40 : memref<?x32x1xf16>
                  %114 = loom.broadcast ins(%113 : tensor<?x?x32x1xf16>) outs(%107 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %103 : memref<?x?x32x1xf16>
                  %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%97, %114 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%107 : tensor<?x?x32x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %121 = arith.mulf %in, %in_3 : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x?x32x128xf16>
                  loom.semaphore_give %96 : memref<?x?x32x128xf16>
                  %116 = linalg.fill ins(%cst : f16) outs(%101 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
                  %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%115 : tensor<?x?x32x128xf16>) outs(%116 : tensor<?x32x128xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %121 = arith.addf %in, %out : f16
                    linalg.yield %121 : f16
                  } -> tensor<?x32x128xf16>
                  loom.semaphore_give %106 : memref<?x?x32x128xf16>
                  %118 = loom.subview %arg3[%28, 0, 0] [%21, 32, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  %119 = loom.bufferize_to_memref %117 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
                  %120 = arith.addi %arg5, %91 : index
                  loom.copy %119, %118 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%120, %arg6], LR : [%120, %arg6]) : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
                  loom.semaphore_give %100 : memref<?x32x128xf16>
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
