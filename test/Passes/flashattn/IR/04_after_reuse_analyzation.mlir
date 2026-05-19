module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y1y8__d0i1_d1i1_d2i0__f01(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %18 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %19 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %21 = arith.ceildivui %c32, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
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
                %28 = arith.muli %26, %19 : index
                %29 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %31 = loom.subview %arg2[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %32 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %33 = arith.ceildivui %c4096, %20 : index
                %34 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %36 = loom.init_tensor %35[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %38 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %39 = loom.semaphore_take %38 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %40 = loom.init_tensor %39[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %42 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %44 = loom.init_tensor %43[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %46 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %47 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %48 = loom.init_tensor %47[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %49 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %56 = loom.init_tensor %55[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %57 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %59 = loom.init_tensor %58[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %60 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %62 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %64 = loom.init_tensor %63[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %65 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %67 = loom.init_tensor %66[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %68 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %69 = loom.semaphore_take %68 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %70:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %45, %arg11 = %41, %arg12 = %37) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %78 = arith.muli %arg9, %20 : index
                  %79 = loom.subview %arg0[%27, 0, %78] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %79, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %80 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %81 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %82 = linalg.batch_matmul ins(%32, %80 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%81 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %83 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%82 : tensor<?x?x?xf16>) outs(%83 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.maximumf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%53 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in_3, %cst_2 : f16
                    %99 = arith.cmpf ogt, %in, %98 : f16
                    %100 = arith.select %99, %in, %98 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = loom.broadcast ins(%85 : tensor<?x?x1xf16>) outs(%67 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %86 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in, %cst_2 : f16
                    %99 = arith.subf %98, %in_3 : f16
                    %100 = math.exp %99 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x?x?xf16>
                  %88 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%87 : tensor<?x?x?xf16>) outs(%88 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.addf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %85 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%59 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.subf %in, %in_3 : f16
                    %99 = math.exp %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %90, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in, %in_3 : f16
                    %99 = arith.addf %98, %in_4 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %55 : memref<?x?x1xf16>
                  %92 = loom.subview %arg1[%27, %78, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %92, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %93 = loom.bufferize_to_tensor %69[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %94 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %95 = linalg.batch_matmul ins(%87, %93 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%94 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x?xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %arg12, %90 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in_3, %in_4 : f16
                    %99 = arith.addf %in, %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %58 : memref<?x?x1xf16>
                  loom.semaphore_give %49 : memref<?x?x128xf16>
                  %97 = linalg.copy ins(%85 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %52 : memref<?x?x1xf16>
                  scf.yield %97, %91, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %43 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %71 = loom.broadcast ins(%70#1 : tensor<?x?x1xf16>) outs(%48 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %39 : memref<?x?x1xf16>
                %72 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %74 = loom.init_tensor %73[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70#2, %71 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%74 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %78 = arith.divf %in, %in_3 : f16
                  linalg.yield %78 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %47 : memref<?x?x128xf16>
                loom.semaphore_give %35 : memref<?x?x128xf16>
                %76 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %77 = loom.bufferize_to_memref %75 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %73 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y2y4__d0i1_d1i1_d2i0__f01(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %18 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %19 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %21 = arith.ceildivui %c32, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
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
                %28 = arith.muli %26, %19 : index
                %29 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %31 = loom.subview %arg2[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %32 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %33 = arith.ceildivui %c4096, %20 : index
                %34 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %36 = loom.init_tensor %35[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %38 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %39 = loom.semaphore_take %38 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %40 = loom.init_tensor %39[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %42 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %44 = loom.init_tensor %43[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %46 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %47 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %48 = loom.init_tensor %47[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %49 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %56 = loom.init_tensor %55[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %57 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %59 = loom.init_tensor %58[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %60 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %62 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %64 = loom.init_tensor %63[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %65 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %67 = loom.init_tensor %66[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %68 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %69 = loom.semaphore_take %68 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %70:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %45, %arg11 = %41, %arg12 = %37) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %78 = arith.muli %arg9, %20 : index
                  %79 = loom.subview %arg0[%27, 0, %78] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %79, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %80 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %81 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %82 = linalg.batch_matmul ins(%32, %80 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%81 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %83 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%82 : tensor<?x?x?xf16>) outs(%83 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.maximumf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%53 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in_3, %cst_2 : f16
                    %99 = arith.cmpf ogt, %in, %98 : f16
                    %100 = arith.select %99, %in, %98 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = loom.broadcast ins(%85 : tensor<?x?x1xf16>) outs(%67 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %86 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in, %cst_2 : f16
                    %99 = arith.subf %98, %in_3 : f16
                    %100 = math.exp %99 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x?x?xf16>
                  %88 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%87 : tensor<?x?x?xf16>) outs(%88 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.addf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %85 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%59 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.subf %in, %in_3 : f16
                    %99 = math.exp %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %90, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in, %in_3 : f16
                    %99 = arith.addf %98, %in_4 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %55 : memref<?x?x1xf16>
                  %92 = loom.subview %arg1[%27, %78, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %92, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %93 = loom.bufferize_to_tensor %69[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %94 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %95 = linalg.batch_matmul ins(%87, %93 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%94 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x?xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %arg12, %90 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in_3, %in_4 : f16
                    %99 = arith.addf %in, %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %58 : memref<?x?x1xf16>
                  loom.semaphore_give %49 : memref<?x?x128xf16>
                  %97 = linalg.copy ins(%85 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %52 : memref<?x?x1xf16>
                  scf.yield %97, %91, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %43 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %71 = loom.broadcast ins(%70#1 : tensor<?x?x1xf16>) outs(%48 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %39 : memref<?x?x1xf16>
                %72 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %74 = loom.init_tensor %73[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70#2, %71 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%74 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %78 = arith.divf %in, %in_3 : f16
                  linalg.yield %78 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %47 : memref<?x?x128xf16>
                loom.semaphore_give %35 : memref<?x?x128xf16>
                %76 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %77 = loom.bufferize_to_memref %75 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %73 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y4y2__d0i1_d1i1_d2i0__f01(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %18 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %19 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %21 = arith.ceildivui %c32, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
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
                %28 = arith.muli %26, %19 : index
                %29 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %31 = loom.subview %arg2[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %32 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %33 = arith.ceildivui %c4096, %20 : index
                %34 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %36 = loom.init_tensor %35[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %38 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %39 = loom.semaphore_take %38 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %40 = loom.init_tensor %39[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %42 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %44 = loom.init_tensor %43[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %46 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %47 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %48 = loom.init_tensor %47[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %49 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %56 = loom.init_tensor %55[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %57 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %59 = loom.init_tensor %58[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %60 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %62 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %64 = loom.init_tensor %63[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %65 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %67 = loom.init_tensor %66[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %68 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %69 = loom.semaphore_take %68 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %70:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %45, %arg11 = %41, %arg12 = %37) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %78 = arith.muli %arg9, %20 : index
                  %79 = loom.subview %arg0[%27, 0, %78] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %79, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %80 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %81 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %82 = linalg.batch_matmul ins(%32, %80 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%81 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %83 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%82 : tensor<?x?x?xf16>) outs(%83 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.maximumf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%53 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in_3, %cst_2 : f16
                    %99 = arith.cmpf ogt, %in, %98 : f16
                    %100 = arith.select %99, %in, %98 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = loom.broadcast ins(%85 : tensor<?x?x1xf16>) outs(%67 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %86 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in, %cst_2 : f16
                    %99 = arith.subf %98, %in_3 : f16
                    %100 = math.exp %99 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x?x?xf16>
                  %88 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%87 : tensor<?x?x?xf16>) outs(%88 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.addf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %85 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%59 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.subf %in, %in_3 : f16
                    %99 = math.exp %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %90, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in, %in_3 : f16
                    %99 = arith.addf %98, %in_4 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %55 : memref<?x?x1xf16>
                  %92 = loom.subview %arg1[%27, %78, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %92, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %93 = loom.bufferize_to_tensor %69[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %94 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %95 = linalg.batch_matmul ins(%87, %93 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%94 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x?xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %arg12, %90 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in_3, %in_4 : f16
                    %99 = arith.addf %in, %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %58 : memref<?x?x1xf16>
                  loom.semaphore_give %49 : memref<?x?x128xf16>
                  %97 = linalg.copy ins(%85 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %52 : memref<?x?x1xf16>
                  scf.yield %97, %91, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %43 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %71 = loom.broadcast ins(%70#1 : tensor<?x?x1xf16>) outs(%48 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %39 : memref<?x?x1xf16>
                %72 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %74 = loom.init_tensor %73[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70#2, %71 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%74 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %78 = arith.divf %in, %in_3 : f16
                  linalg.yield %78 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %47 : memref<?x?x128xf16>
                loom.semaphore_give %35 : memref<?x?x128xf16>
                %76 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %77 = loom.bufferize_to_memref %75 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %73 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y8y1__d0i1_d1i1_d2i0__f01(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %18 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %19 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %21 = arith.ceildivui %c32, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %21 step %c1 {
              %23 = arith.ceildivui %22, %c64 : index
              scf.for %arg8 = %c0 to %23 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %25 = arith.muli %arg7, %18 : index
                %26 = arith.muli %24, %19 : index
                %27 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %28 = loom.semaphore_take %27 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %29 = loom.subview %arg2[%25, %26, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %29, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %30 = loom.bufferize_to_tensor %28[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %31 = arith.ceildivui %c4096, %20 : index
                %32 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %33 = loom.semaphore_take %32 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %34 = loom.init_tensor %33[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %35 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %36 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %37 = loom.semaphore_take %36 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %38 = loom.init_tensor %37[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %39 = linalg.fill ins(%cst_0 : f16) outs(%38 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %40 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %42 = loom.init_tensor %41[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %43 = linalg.fill ins(%cst_1 : f16) outs(%42 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %44 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %45 = loom.semaphore_take %44 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %46 = loom.init_tensor %45[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %47 = loom.semaphore_take %44 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %48 = loom.init_tensor %47[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %49 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %50 = loom.semaphore_take %49 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %51 = loom.init_tensor %50[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %52 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %53 = loom.semaphore_take %52 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %54 = loom.init_tensor %53[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %55 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %57 = loom.init_tensor %56[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %58 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %59 = loom.semaphore_take %58 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %60 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %62 = loom.init_tensor %61[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %63 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %64 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %65 = loom.init_tensor %64[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %66 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %67 = loom.semaphore_take %66 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %68:3 = scf.for %arg9 = %c0 to %31 step %c1 iter_args(%arg10 = %43, %arg11 = %39, %arg12 = %35) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %76 = arith.muli %arg9, %20 : index
                  %77 = loom.subview %arg0[%25, 0, %76] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %77, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %78 = loom.bufferize_to_tensor %59[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %79 = linalg.fill ins(%cst : f16) outs(%62 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %80 = linalg.batch_matmul ins(%30, %78 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%79 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %59 : memref<?x128x?xf16>
                  %81 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%80 : tensor<?x?x?xf16>) outs(%81 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %96 = arith.maximumf %in, %out : f16
                    linalg.yield %96 : f16
                  } -> tensor<?x?x1xf16>
                  %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %82 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%51 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %96 = arith.mulf %in_3, %cst_2 : f16
                    %97 = arith.cmpf ogt, %in, %96 : f16
                    %98 = arith.select %97, %in, %96 : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %84 = loom.broadcast ins(%83 : tensor<?x?x1xf16>) outs(%65 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80, %84 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%62 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %96 = arith.mulf %in, %cst_2 : f16
                    %97 = arith.subf %96, %in_3 : f16
                    %98 = math.exp %97 : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %64 : memref<?x?x?xf16>
                  %86 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<?x?x?xf16>) outs(%86 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %96 = arith.addf %in, %out : f16
                    linalg.yield %96 : f16
                  } -> tensor<?x?x1xf16>
                  %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %83 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%57 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %96 = arith.subf %in, %in_3 : f16
                    %97 = math.exp %96 : f16
                    linalg.yield %97 : f16
                  } -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %88, %87 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %96 = arith.mulf %in, %in_3 : f16
                    %97 = arith.addf %96, %in_4 : f16
                    linalg.yield %97 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %53 : memref<?x?x1xf16>
                  %90 = loom.subview %arg1[%25, %76, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %90, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %91 = loom.bufferize_to_tensor %67[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %92 = linalg.fill ins(%cst : f16) outs(%48 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %93 = linalg.batch_matmul ins(%85, %91 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%92 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x?x128xf16>
                  loom.semaphore_give %61 : memref<?x?x?xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93, %arg12, %88 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %96 = arith.mulf %in_3, %in_4 : f16
                    %97 = arith.addf %in, %96 : f16
                    linalg.yield %97 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %56 : memref<?x?x1xf16>
                  loom.semaphore_give %47 : memref<?x?x128xf16>
                  %95 = linalg.copy ins(%83 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %50 : memref<?x?x1xf16>
                  scf.yield %95, %89, %94 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %41 : memref<?x?x1xf16>
                loom.semaphore_give %28 : memref<?x?x128xf16>
                %69 = loom.broadcast ins(%68#1 : tensor<?x?x1xf16>) outs(%46 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %37 : memref<?x?x1xf16>
                %70 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %71 = loom.semaphore_take %70 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %72 = loom.init_tensor %71[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%68#2, %69 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%72 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %76 = arith.divf %in, %in_3 : f16
                  linalg.yield %76 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %45 : memref<?x?x128xf16>
                loom.semaphore_give %33 : memref<?x?x128xf16>
                %74 = loom.subview %arg3[%25, %26, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %75 = loom.bufferize_to_memref %73 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %71 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x1x8_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %18 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %19 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %21 = arith.ceildivui %c32, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
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
                %28 = arith.muli %26, %19 : index
                %29 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %31 = loom.subview %arg2[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %32 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %33 = arith.ceildivui %c4096, %20 : index
                %34 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %36 = loom.init_tensor %35[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %38 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %39 = loom.semaphore_take %38 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %40 = loom.init_tensor %39[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %42 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %44 = loom.init_tensor %43[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %46 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %47 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %48 = loom.init_tensor %47[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %49 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %56 = loom.init_tensor %55[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %57 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %59 = loom.init_tensor %58[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %60 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %62 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %64 = loom.init_tensor %63[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %65 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %67 = loom.init_tensor %66[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %68 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %69 = loom.semaphore_take %68 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %70:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %45, %arg11 = %41, %arg12 = %37) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %78 = arith.muli %arg9, %20 : index
                  %79 = loom.subview %arg0[%27, 0, %78] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %79, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %80 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %81 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %82 = linalg.batch_matmul ins(%32, %80 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%81 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %83 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%82 : tensor<?x?x?xf16>) outs(%83 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.maximumf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%53 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in_3, %cst_2 : f16
                    %99 = arith.cmpf ogt, %in, %98 : f16
                    %100 = arith.select %99, %in, %98 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = loom.broadcast ins(%85 : tensor<?x?x1xf16>) outs(%67 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %86 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in, %cst_2 : f16
                    %99 = arith.subf %98, %in_3 : f16
                    %100 = math.exp %99 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x?x?xf16>
                  %88 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%87 : tensor<?x?x?xf16>) outs(%88 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.addf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %85 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%59 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.subf %in, %in_3 : f16
                    %99 = math.exp %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %90, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in, %in_3 : f16
                    %99 = arith.addf %98, %in_4 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %55 : memref<?x?x1xf16>
                  %92 = loom.subview %arg1[%27, %78, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %92, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %93 = loom.bufferize_to_tensor %69[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %94 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %95 = linalg.batch_matmul ins(%87, %93 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%94 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x?xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %arg12, %90 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in_3, %in_4 : f16
                    %99 = arith.addf %in, %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %58 : memref<?x?x1xf16>
                  loom.semaphore_give %49 : memref<?x?x128xf16>
                  %97 = linalg.copy ins(%85 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %52 : memref<?x?x1xf16>
                  scf.yield %97, %91, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %43 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %71 = loom.broadcast ins(%70#1 : tensor<?x?x1xf16>) outs(%48 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %39 : memref<?x?x1xf16>
                %72 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %74 = loom.init_tensor %73[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70#2, %71 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%74 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %78 = arith.divf %in, %in_3 : f16
                  linalg.yield %78 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %47 : memref<?x?x128xf16>
                loom.semaphore_give %35 : memref<?x?x128xf16>
                %76 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %77 = loom.bufferize_to_memref %75 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %73 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x2x4_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %18 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %19 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %21 = arith.ceildivui %c32, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
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
                %28 = arith.muli %26, %19 : index
                %29 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %31 = loom.subview %arg2[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %32 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %33 = arith.ceildivui %c4096, %20 : index
                %34 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %36 = loom.init_tensor %35[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %38 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %39 = loom.semaphore_take %38 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %40 = loom.init_tensor %39[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %42 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %44 = loom.init_tensor %43[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %46 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %47 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %48 = loom.init_tensor %47[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %49 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %56 = loom.init_tensor %55[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %57 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %59 = loom.init_tensor %58[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %60 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %62 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %64 = loom.init_tensor %63[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %65 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %67 = loom.init_tensor %66[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %68 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %69 = loom.semaphore_take %68 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %70:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %45, %arg11 = %41, %arg12 = %37) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %78 = arith.muli %arg9, %20 : index
                  %79 = loom.subview %arg0[%27, 0, %78] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %79, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %80 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %81 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %82 = linalg.batch_matmul ins(%32, %80 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%81 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %83 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%82 : tensor<?x?x?xf16>) outs(%83 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.maximumf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%53 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in_3, %cst_2 : f16
                    %99 = arith.cmpf ogt, %in, %98 : f16
                    %100 = arith.select %99, %in, %98 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = loom.broadcast ins(%85 : tensor<?x?x1xf16>) outs(%67 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %86 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in, %cst_2 : f16
                    %99 = arith.subf %98, %in_3 : f16
                    %100 = math.exp %99 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x?x?xf16>
                  %88 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%87 : tensor<?x?x?xf16>) outs(%88 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.addf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %85 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%59 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.subf %in, %in_3 : f16
                    %99 = math.exp %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %90, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in, %in_3 : f16
                    %99 = arith.addf %98, %in_4 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %55 : memref<?x?x1xf16>
                  %92 = loom.subview %arg1[%27, %78, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %92, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %93 = loom.bufferize_to_tensor %69[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %94 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %95 = linalg.batch_matmul ins(%87, %93 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%94 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x?xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %arg12, %90 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in_3, %in_4 : f16
                    %99 = arith.addf %in, %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %58 : memref<?x?x1xf16>
                  loom.semaphore_give %49 : memref<?x?x128xf16>
                  %97 = linalg.copy ins(%85 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %52 : memref<?x?x1xf16>
                  scf.yield %97, %91, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %43 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %71 = loom.broadcast ins(%70#1 : tensor<?x?x1xf16>) outs(%48 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %39 : memref<?x?x1xf16>
                %72 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %74 = loom.init_tensor %73[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70#2, %71 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%74 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %78 = arith.divf %in, %in_3 : f16
                  linalg.yield %78 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %47 : memref<?x?x128xf16>
                loom.semaphore_give %35 : memref<?x?x128xf16>
                %76 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %77 = loom.bufferize_to_memref %75 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %73 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x4x2_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %18 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %19 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %21 = arith.ceildivui %c32, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
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
                %28 = arith.muli %26, %19 : index
                %29 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %31 = loom.subview %arg2[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %32 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %33 = arith.ceildivui %c4096, %20 : index
                %34 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %35 = loom.semaphore_take %34 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %36 = loom.init_tensor %35[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %38 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %39 = loom.semaphore_take %38 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %40 = loom.init_tensor %39[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %42 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %43 = loom.semaphore_take %42 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %44 = loom.init_tensor %43[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %46 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %47 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %48 = loom.init_tensor %47[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %49 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %55 = loom.semaphore_take %54 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %56 = loom.init_tensor %55[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %57 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %58 = loom.semaphore_take %57 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %59 = loom.init_tensor %58[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %60 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %62 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %64 = loom.init_tensor %63[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %65 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %67 = loom.init_tensor %66[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %68 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %69 = loom.semaphore_take %68 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %70:3 = scf.for %arg9 = %c0 to %33 step %c1 iter_args(%arg10 = %45, %arg11 = %41, %arg12 = %37) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %78 = arith.muli %arg9, %20 : index
                  %79 = loom.subview %arg0[%27, 0, %78] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %79, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %80 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %81 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %82 = linalg.batch_matmul ins(%32, %80 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%81 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %83 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%82 : tensor<?x?x?xf16>) outs(%83 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.maximumf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%53 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in_3, %cst_2 : f16
                    %99 = arith.cmpf ogt, %in, %98 : f16
                    %100 = arith.select %99, %in, %98 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = loom.broadcast ins(%85 : tensor<?x?x1xf16>) outs(%67 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %86 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.mulf %in, %cst_2 : f16
                    %99 = arith.subf %98, %in_3 : f16
                    %100 = math.exp %99 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x?x?xf16>
                  %88 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%87 : tensor<?x?x?xf16>) outs(%88 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %98 = arith.addf %in, %out : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %85 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%59 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %98 = arith.subf %in, %in_3 : f16
                    %99 = math.exp %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %90, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in, %in_3 : f16
                    %99 = arith.addf %98, %in_4 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %55 : memref<?x?x1xf16>
                  %92 = loom.subview %arg1[%27, %78, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %92, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %93 = loom.bufferize_to_tensor %69[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %94 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %95 = linalg.batch_matmul ins(%87, %93 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%94 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x?xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %arg12, %90 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %98 = arith.mulf %in_3, %in_4 : f16
                    %99 = arith.addf %in, %98 : f16
                    linalg.yield %99 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %58 : memref<?x?x1xf16>
                  loom.semaphore_give %49 : memref<?x?x128xf16>
                  %97 = linalg.copy ins(%85 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %52 : memref<?x?x1xf16>
                  scf.yield %97, %91, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %43 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %71 = loom.broadcast ins(%70#1 : tensor<?x?x1xf16>) outs(%48 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %39 : memref<?x?x1xf16>
                %72 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %74 = loom.init_tensor %73[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70#2, %71 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%74 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %78 = arith.divf %in, %in_3 : f16
                  linalg.yield %78 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %47 : memref<?x?x128xf16>
                loom.semaphore_give %35 : memref<?x?x128xf16>
                %76 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %77 = loom.bufferize_to_memref %75 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %73 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8x1_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %18 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %19 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %21 = arith.ceildivui %c32, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %21 step %c1 {
              %23 = arith.ceildivui %22, %c64 : index
              scf.for %arg8 = %c0 to %23 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %25 = arith.muli %arg7, %18 : index
                %26 = arith.muli %24, %19 : index
                %27 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %28 = loom.semaphore_take %27 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %29 = loom.subview %arg2[%25, %26, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %29, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %30 = loom.bufferize_to_tensor %28[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %31 = arith.ceildivui %c4096, %20 : index
                %32 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %33 = loom.semaphore_take %32 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %34 = loom.init_tensor %33[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %35 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %36 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %37 = loom.semaphore_take %36 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %38 = loom.init_tensor %37[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %39 = linalg.fill ins(%cst_0 : f16) outs(%38 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %40 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %42 = loom.init_tensor %41[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %43 = linalg.fill ins(%cst_1 : f16) outs(%42 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %44 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %45 = loom.semaphore_take %44 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %46 = loom.init_tensor %45[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %47 = loom.semaphore_take %44 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %48 = loom.init_tensor %47[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %49 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %50 = loom.semaphore_take %49 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %51 = loom.init_tensor %50[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %52 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %53 = loom.semaphore_take %52 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %54 = loom.init_tensor %53[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %55 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %57 = loom.init_tensor %56[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %58 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %59 = loom.semaphore_take %58 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %60 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %61 = loom.semaphore_take %60 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %62 = loom.init_tensor %61[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %63 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %64 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %65 = loom.init_tensor %64[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %66 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %67 = loom.semaphore_take %66 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %68:3 = scf.for %arg9 = %c0 to %31 step %c1 iter_args(%arg10 = %43, %arg11 = %39, %arg12 = %35) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %76 = arith.muli %arg9, %20 : index
                  %77 = loom.subview %arg0[%25, 0, %76] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %77, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %78 = loom.bufferize_to_tensor %59[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %79 = linalg.fill ins(%cst : f16) outs(%62 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %80 = linalg.batch_matmul ins(%30, %78 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%79 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %59 : memref<?x128x?xf16>
                  %81 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%80 : tensor<?x?x?xf16>) outs(%81 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %96 = arith.maximumf %in, %out : f16
                    linalg.yield %96 : f16
                  } -> tensor<?x?x1xf16>
                  %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %82 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%51 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %96 = arith.mulf %in_3, %cst_2 : f16
                    %97 = arith.cmpf ogt, %in, %96 : f16
                    %98 = arith.select %97, %in, %96 : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x1xf16>
                  %84 = loom.broadcast ins(%83 : tensor<?x?x1xf16>) outs(%65 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80, %84 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%62 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %96 = arith.mulf %in, %cst_2 : f16
                    %97 = arith.subf %96, %in_3 : f16
                    %98 = math.exp %97 : f16
                    linalg.yield %98 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %64 : memref<?x?x?xf16>
                  %86 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<?x?x?xf16>) outs(%86 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %96 = arith.addf %in, %out : f16
                    linalg.yield %96 : f16
                  } -> tensor<?x?x1xf16>
                  %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %83 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%57 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %96 = arith.subf %in, %in_3 : f16
                    %97 = math.exp %96 : f16
                    linalg.yield %97 : f16
                  } -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %88, %87 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %96 = arith.mulf %in, %in_3 : f16
                    %97 = arith.addf %96, %in_4 : f16
                    linalg.yield %97 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %53 : memref<?x?x1xf16>
                  %90 = loom.subview %arg1[%25, %76, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %90, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %91 = loom.bufferize_to_tensor %67[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %92 = linalg.fill ins(%cst : f16) outs(%48 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %93 = linalg.batch_matmul ins(%85, %91 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%92 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %67 : memref<?x?x128xf16>
                  loom.semaphore_give %61 : memref<?x?x?xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%93, %arg12, %88 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %96 = arith.mulf %in_3, %in_4 : f16
                    %97 = arith.addf %in, %96 : f16
                    linalg.yield %97 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %56 : memref<?x?x1xf16>
                  loom.semaphore_give %47 : memref<?x?x128xf16>
                  %95 = linalg.copy ins(%83 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %50 : memref<?x?x1xf16>
                  scf.yield %95, %89, %94 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %41 : memref<?x?x1xf16>
                loom.semaphore_give %28 : memref<?x?x128xf16>
                %69 = loom.broadcast ins(%68#1 : tensor<?x?x1xf16>) outs(%46 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %37 : memref<?x?x1xf16>
                %70 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %71 = loom.semaphore_take %70 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %72 = loom.init_tensor %71[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%68#2, %69 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%72 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %76 = arith.divf %in, %in_3 : f16
                  linalg.yield %76 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %45 : memref<?x?x128xf16>
                loom.semaphore_give %33 : memref<?x?x128xf16>
                %74 = loom.subview %arg3[%25, %26, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %75 = loom.bufferize_to_memref %73 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %71 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
