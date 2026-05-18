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
    func.func @attention__x8_y1y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_x_level0_bc8_n(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
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
                %32 = arith.addi %arg6, %arg4 : index
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %32], LR : [%arg5, %32]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %33 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = arith.ceildivui %c4096, %20 : index
                %35 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %36 = loom.semaphore_take %35 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %37 = loom.init_tensor %36[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %39 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %40 = loom.semaphore_take %39 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %41 = loom.init_tensor %40[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %42 = linalg.fill ins(%cst_0 : f16) outs(%41 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %43 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %45 = loom.init_tensor %44[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %46 = linalg.fill ins(%cst_1 : f16) outs(%45 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %47 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %48 = loom.semaphore_take %47 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %49 = loom.init_tensor %48[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %50 = loom.semaphore_take %47 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %51 = loom.init_tensor %50[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %52 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %54 = loom.init_tensor %53[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %55 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %57 = loom.init_tensor %56[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %58 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %60 = loom.init_tensor %59[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %61 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %62 = loom.semaphore_take %61 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %63 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %64 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %65 = loom.init_tensor %64[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %66 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %68 = loom.init_tensor %67[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %69 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %70 = loom.semaphore_take %69 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %71:3 = scf.for %arg9 = %c0 to %34 step %c1 iter_args(%arg10 = %46, %arg11 = %42, %arg12 = %38) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %79 = arith.muli %arg9, %20 : index
                  %80 = loom.subview %arg0[%27, 0, %79] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %80, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %32], LR : [%c7, %32]) : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %81 = loom.bufferize_to_tensor %62[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %82 = linalg.fill ins(%cst : f16) outs(%65 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %83 = linalg.batch_matmul ins(%33, %81 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%82 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %62 : memref<?x128x?xf16>
                  %84 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%83 : tensor<?x?x?xf16>) outs(%84 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %101 = arith.maximumf %in, %out : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85 : tensor<?x?x1xf16>) outs(%57 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %101 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %86 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%57 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.cmpf ogt, %in, %in_3 : f16
                    %102 = arith.select %101, %in, %in_3 : f16
                    linalg.yield %102 : f16
                  } -> tensor<?x?x1xf16>
                  %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83 : tensor<?x?x?xf16>) outs(%65 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %101 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x?xf16>
                  %89 = loom.broadcast ins(%87 : tensor<?x?x1xf16>) outs(%68 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %89 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%65 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.subf %in, %in_3 : f16
                    %102 = math.exp %101 : f16
                    linalg.yield %102 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %67 : memref<?x?x?xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %87 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%60 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.subf %in, %in_3 : f16
                    %102 = math.exp %101 : f16
                    linalg.yield %102 : f16
                  } -> tensor<?x?x1xf16>
                  %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %91 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.mulf %in, %in_3 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%90 : tensor<?x?x?xf16>) outs(%92 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %101 = arith.addf %in, %out : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg12, %91 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%51 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.mulf %in, %in_3 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %59 : memref<?x?x1xf16>
                  %95 = loom.subview %arg1[%27, %79, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %95, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %32], LR : [%c7, %32]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %96 = loom.bufferize_to_tensor %70[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %97 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %98 = linalg.batch_matmul ins(%90, %96 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%97 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %70 : memref<?x?x128xf16>
                  loom.semaphore_give %64 : memref<?x?x?xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%98, %94 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.addf %in, %in_3 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %53 : memref<?x?x128xf16>
                  loom.semaphore_give %50 : memref<?x?x128xf16>
                  %100 = linalg.copy ins(%87 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %56 : memref<?x?x1xf16>
                  scf.yield %100, %93, %99 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %44 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %72 = loom.broadcast ins(%71#1 : tensor<?x?x1xf16>) outs(%49 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %40 : memref<?x?x1xf16>
                %73 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %74 = loom.semaphore_take %73 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %75 = loom.init_tensor %74[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71#2, %72 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%75 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %79 = arith.divf %in, %in_3 : f16
                  linalg.yield %79 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %48 : memref<?x?x128xf16>
                loom.semaphore_give %36 : memref<?x?x128xf16>
                %77 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %78 = loom.bufferize_to_memref %76 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %32], LR : [%arg5, %32]) : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %74 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y2y4__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc2_dim_x_level0_bc8_dim_y_level0_bc2_n(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
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
                %32 = arith.muli %arg4, %c2 : index
                %33 = arith.addi %arg6, %32 : index
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %34 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %35 = arith.ceildivui %c4096, %20 : index
                %36 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %40 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %42 = loom.init_tensor %41[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %44 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %45 = loom.semaphore_take %44 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %46 = loom.init_tensor %45[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %48 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %49 = loom.semaphore_take %48 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.semaphore_take %48 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %52 = loom.init_tensor %51[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %53 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %54 = loom.semaphore_take %53 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %55 = loom.init_tensor %54[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %56 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %57 = loom.semaphore_take %56 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %58 = loom.init_tensor %57[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %59 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %61 = loom.init_tensor %60[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %62 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %64 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %66 = loom.init_tensor %65[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %67 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %69 = loom.init_tensor %68[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %70 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %71 = loom.semaphore_take %70 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %72:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %47, %arg11 = %43, %arg12 = %39) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %80 = arith.muli %arg9, %20 : index
                  %81 = loom.subview %arg0[%27, 0, %80] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  %82 = arith.addi %32, %c1 : index
                  loom.copy %81, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %32], LR : [%c7, %82]) : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %83 = loom.bufferize_to_tensor %63[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %84 = linalg.fill ins(%cst : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %85 = linalg.batch_matmul ins(%34, %83 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%84 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %63 : memref<?x128x?xf16>
                  %86 = linalg.fill ins(%cst_1 : f16) outs(%58 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<?x?x?xf16>) outs(%86 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.maximumf %in, %out : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87 : tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %88 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.cmpf ogt, %in, %in_3 : f16
                    %104 = arith.select %103, %in, %in_3 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85 : tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x?xf16>
                  %91 = loom.broadcast ins(%89 : tensor<?x?x1xf16>) outs(%69 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %91 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.subf %in, %in_3 : f16
                    %104 = math.exp %103 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x?x?xf16>
                  %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%61 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.subf %in, %in_3 : f16
                    %104 = math.exp %103 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x1xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %93 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.mulf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<?x?x?xf16>) outs(%94 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.addf %in, %out : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg12, %93 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%52 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.mulf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %60 : memref<?x?x1xf16>
                  %97 = loom.subview %arg1[%27, %80, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %97, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %32], LR : [%c7, %82]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %98 = loom.bufferize_to_tensor %71[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %99 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %100 = linalg.batch_matmul ins(%92, %98 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%99 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %71 : memref<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x?xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100, %96 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.addf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %54 : memref<?x?x128xf16>
                  loom.semaphore_give %51 : memref<?x?x128xf16>
                  %102 = linalg.copy ins(%89 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %57 : memref<?x?x1xf16>
                  scf.yield %102, %95, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %45 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %73 = loom.broadcast ins(%72#1 : tensor<?x?x1xf16>) outs(%50 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %41 : memref<?x?x1xf16>
                %74 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %76 = loom.init_tensor %75[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72#2, %73 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%76 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %80 = arith.divf %in, %in_3 : f16
                  linalg.yield %80 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %49 : memref<?x?x128xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %78 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %79 = loom.bufferize_to_memref %77 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %79, %78 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %75 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y4y2__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc4_dim_x_level0_bc8_dim_y_level0_bc4_n(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
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
                %32 = arith.muli %arg4, %c4 : index
                %33 = arith.addi %arg6, %32 : index
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %34 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %35 = arith.ceildivui %c4096, %20 : index
                %36 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %40 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %42 = loom.init_tensor %41[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %44 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %45 = loom.semaphore_take %44 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %46 = loom.init_tensor %45[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %48 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %49 = loom.semaphore_take %48 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.semaphore_take %48 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %52 = loom.init_tensor %51[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %53 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %54 = loom.semaphore_take %53 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %55 = loom.init_tensor %54[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %56 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %57 = loom.semaphore_take %56 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %58 = loom.init_tensor %57[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %59 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %61 = loom.init_tensor %60[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %62 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %64 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %66 = loom.init_tensor %65[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %67 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %69 = loom.init_tensor %68[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %70 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %71 = loom.semaphore_take %70 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %72:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %47, %arg11 = %43, %arg12 = %39) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %80 = arith.muli %arg9, %20 : index
                  %81 = loom.subview %arg0[%27, 0, %80] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  %82 = arith.addi %32, %c3 : index
                  loom.copy %81, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %82]) : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %83 = loom.bufferize_to_tensor %63[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %84 = linalg.fill ins(%cst : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %85 = linalg.batch_matmul ins(%34, %83 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%84 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %63 : memref<?x128x?xf16>
                  %86 = linalg.fill ins(%cst_1 : f16) outs(%58 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<?x?x?xf16>) outs(%86 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.maximumf %in, %out : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87 : tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %88 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.cmpf ogt, %in, %in_3 : f16
                    %104 = arith.select %103, %in, %in_3 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85 : tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x?xf16>
                  %91 = loom.broadcast ins(%89 : tensor<?x?x1xf16>) outs(%69 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %91 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.subf %in, %in_3 : f16
                    %104 = math.exp %103 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x?x?xf16>
                  %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%61 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.subf %in, %in_3 : f16
                    %104 = math.exp %103 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x1xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %93 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.mulf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<?x?x?xf16>) outs(%94 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.addf %in, %out : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg12, %93 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%52 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.mulf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %60 : memref<?x?x1xf16>
                  %97 = loom.subview %arg1[%27, %80, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %97, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %82]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %98 = loom.bufferize_to_tensor %71[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %99 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %100 = linalg.batch_matmul ins(%92, %98 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%99 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %71 : memref<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x?xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100, %96 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.addf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %54 : memref<?x?x128xf16>
                  loom.semaphore_give %51 : memref<?x?x128xf16>
                  %102 = linalg.copy ins(%89 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %57 : memref<?x?x1xf16>
                  scf.yield %102, %95, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %45 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %73 = loom.broadcast ins(%72#1 : tensor<?x?x1xf16>) outs(%50 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %41 : memref<?x?x1xf16>
                %74 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %76 = loom.init_tensor %75[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72#2, %73 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%76 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %80 = arith.divf %in, %in_3 : f16
                  linalg.yield %80 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %49 : memref<?x?x128xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %78 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %79 = loom.bufferize_to_memref %77 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %79, %78 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %75 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y8y1__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level1_bc8_dim_x_level0_bc8_dim_y_level1_bc8_n(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
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
                %30 = arith.muli %arg4, %c8 : index
                %31 = arith.addi %arg6, %30 : index
                loom.copy %29, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %32 = loom.bufferize_to_tensor %28[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
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
                %51 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %53 = loom.init_tensor %52[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
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
                  %79 = loom.subview %arg0[%25, 0, %78] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %79, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %80 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %81 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %82 = linalg.batch_matmul ins(%32, %80 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%81 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %83 = linalg.fill ins(%cst_1 : f16) outs(%56 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%82 : tensor<?x?x?xf16>) outs(%83 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %100 = arith.maximumf %in, %out : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84 : tensor<?x?x1xf16>) outs(%56 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %100 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %85 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%56 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.cmpf ogt, %in, %in_3 : f16
                    %101 = arith.select %100, %in, %in_3 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82 : tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %100 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x?xf16>
                  %88 = loom.broadcast ins(%86 : tensor<?x?x1xf16>) outs(%67 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %88 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.subf %in, %in_3 : f16
                    %101 = math.exp %100 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x?x?xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %86 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%59 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.subf %in, %in_3 : f16
                    %101 = math.exp %100 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %90 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.mulf %in, %in_3 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%89 : tensor<?x?x?xf16>) outs(%91 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %100 = arith.addf %in, %out : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg12, %90 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%50 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.mulf %in, %in_3 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %58 : memref<?x?x1xf16>
                  %94 = loom.subview %arg1[%25, %78, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %94, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %95 = loom.bufferize_to_tensor %69[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %96 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %97 = linalg.batch_matmul ins(%89, %95 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%96 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x?xf16>
                  %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%97, %93 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.addf %in, %in_3 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %52 : memref<?x?x128xf16>
                  loom.semaphore_give %49 : memref<?x?x128xf16>
                  %99 = linalg.copy ins(%86 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %55 : memref<?x?x1xf16>
                  scf.yield %99, %92, %98 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %43 : memref<?x?x1xf16>
                loom.semaphore_give %28 : memref<?x?x128xf16>
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
                %76 = loom.subview %arg3[%25, %26, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %77 = loom.bufferize_to_memref %75 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %31], LR : [%arg5, %31]) : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
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
    func.func @attention__x1x8_y8__d0i1_d1i1_d2i0__f01__n_dim_y_level0_bc8_dim_y_level0_bc8_n(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
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
                %32 = arith.addi %arg5, %arg4 : index
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%32, %arg6], LR : [%32, %arg6]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %33 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = arith.ceildivui %c4096, %20 : index
                %35 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %36 = loom.semaphore_take %35 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %37 = loom.init_tensor %36[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %38 = linalg.fill ins(%cst : f16) outs(%37 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %39 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %40 = loom.semaphore_take %39 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %41 = loom.init_tensor %40[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %42 = linalg.fill ins(%cst_0 : f16) outs(%41 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %43 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %44 = loom.semaphore_take %43 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %45 = loom.init_tensor %44[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %46 = linalg.fill ins(%cst_1 : f16) outs(%45 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %47 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %48 = loom.semaphore_take %47 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %49 = loom.init_tensor %48[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %50 = loom.semaphore_take %47 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %51 = loom.init_tensor %50[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %52 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %53 = loom.semaphore_take %52 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %54 = loom.init_tensor %53[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %55 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %57 = loom.init_tensor %56[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %58 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %60 = loom.init_tensor %59[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %61 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %62 = loom.semaphore_take %61 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %63 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %64 = loom.semaphore_take %63 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %65 = loom.init_tensor %64[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %66 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %67 = loom.semaphore_take %66 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %68 = loom.init_tensor %67[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %69 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %70 = loom.semaphore_take %69 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %71:3 = scf.for %arg9 = %c0 to %34 step %c1 iter_args(%arg10 = %46, %arg11 = %42, %arg12 = %38) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %79 = arith.muli %arg9, %20 : index
                  %80 = loom.subview %arg0[%27, 0, %79] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %80, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%32, %c0], LR : [%32, %c7]) : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %81 = loom.bufferize_to_tensor %62[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %82 = linalg.fill ins(%cst : f16) outs(%65 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %83 = linalg.batch_matmul ins(%33, %81 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%82 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %62 : memref<?x128x?xf16>
                  %84 = linalg.fill ins(%cst_1 : f16) outs(%57 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%83 : tensor<?x?x?xf16>) outs(%84 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %101 = arith.maximumf %in, %out : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85 : tensor<?x?x1xf16>) outs(%57 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %101 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %86 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%57 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.cmpf ogt, %in, %in_3 : f16
                    %102 = arith.select %101, %in, %in_3 : f16
                    linalg.yield %102 : f16
                  } -> tensor<?x?x1xf16>
                  %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83 : tensor<?x?x?xf16>) outs(%65 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %101 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x?xf16>
                  %89 = loom.broadcast ins(%87 : tensor<?x?x1xf16>) outs(%68 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%88, %89 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%65 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.subf %in, %in_3 : f16
                    %102 = math.exp %101 : f16
                    linalg.yield %102 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %67 : memref<?x?x?xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %87 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%60 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.subf %in, %in_3 : f16
                    %102 = math.exp %101 : f16
                    linalg.yield %102 : f16
                  } -> tensor<?x?x1xf16>
                  %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %91 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.mulf %in, %in_3 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%90 : tensor<?x?x?xf16>) outs(%92 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %101 = arith.addf %in, %out : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg12, %91 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%51 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.mulf %in, %in_3 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %59 : memref<?x?x1xf16>
                  %95 = loom.subview %arg1[%27, %79, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %95, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%32, %c0], LR : [%32, %c7]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %96 = loom.bufferize_to_tensor %70[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %97 = linalg.fill ins(%cst : f16) outs(%54 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %98 = linalg.batch_matmul ins(%90, %96 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%97 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %70 : memref<?x?x128xf16>
                  loom.semaphore_give %64 : memref<?x?x?xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%98, %94 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %101 = arith.addf %in, %in_3 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %53 : memref<?x?x128xf16>
                  loom.semaphore_give %50 : memref<?x?x128xf16>
                  %100 = linalg.copy ins(%87 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %56 : memref<?x?x1xf16>
                  scf.yield %100, %93, %99 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %44 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %72 = loom.broadcast ins(%71#1 : tensor<?x?x1xf16>) outs(%49 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %40 : memref<?x?x1xf16>
                %73 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %74 = loom.semaphore_take %73 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %75 = loom.init_tensor %74[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%71#2, %72 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%75 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %79 = arith.divf %in, %in_3 : f16
                  linalg.yield %79 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %48 : memref<?x?x128xf16>
                loom.semaphore_give %36 : memref<?x?x128xf16>
                %77 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %78 = loom.bufferize_to_memref %76 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%32, %arg6], LR : [%32, %arg6]) : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %74 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x2x4_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc2_dim_y_level0_bc8_dim_x_level0_bc2_dim_y_level0_bc8_n(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
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
                %32 = arith.muli %arg4, %c2 : index
                %33 = arith.addi %arg5, %32 : index
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %34 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %35 = arith.ceildivui %c4096, %20 : index
                %36 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %40 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %42 = loom.init_tensor %41[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %44 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %45 = loom.semaphore_take %44 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %46 = loom.init_tensor %45[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %48 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %49 = loom.semaphore_take %48 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.semaphore_take %48 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %52 = loom.init_tensor %51[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %53 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %54 = loom.semaphore_take %53 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %55 = loom.init_tensor %54[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %56 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %57 = loom.semaphore_take %56 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %58 = loom.init_tensor %57[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %59 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %61 = loom.init_tensor %60[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %62 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %64 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %66 = loom.init_tensor %65[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %67 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %69 = loom.init_tensor %68[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %70 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %71 = loom.semaphore_take %70 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %72:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %47, %arg11 = %43, %arg12 = %39) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %80 = arith.muli %arg9, %20 : index
                  %81 = loom.subview %arg0[%27, 0, %80] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  %82 = arith.addi %32, %c1 : index
                  loom.copy %81, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%32, %c0], LR : [%82, %c7]) : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %83 = loom.bufferize_to_tensor %63[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %84 = linalg.fill ins(%cst : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %85 = linalg.batch_matmul ins(%34, %83 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%84 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %63 : memref<?x128x?xf16>
                  %86 = linalg.fill ins(%cst_1 : f16) outs(%58 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<?x?x?xf16>) outs(%86 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.maximumf %in, %out : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87 : tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %88 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.cmpf ogt, %in, %in_3 : f16
                    %104 = arith.select %103, %in, %in_3 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85 : tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x?xf16>
                  %91 = loom.broadcast ins(%89 : tensor<?x?x1xf16>) outs(%69 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %91 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.subf %in, %in_3 : f16
                    %104 = math.exp %103 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x?x?xf16>
                  %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%61 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.subf %in, %in_3 : f16
                    %104 = math.exp %103 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x1xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %93 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.mulf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<?x?x?xf16>) outs(%94 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.addf %in, %out : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg12, %93 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%52 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.mulf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %60 : memref<?x?x1xf16>
                  %97 = loom.subview %arg1[%27, %80, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %97, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%32, %c0], LR : [%82, %c7]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %98 = loom.bufferize_to_tensor %71[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %99 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %100 = linalg.batch_matmul ins(%92, %98 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%99 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %71 : memref<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x?xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100, %96 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.addf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %54 : memref<?x?x128xf16>
                  loom.semaphore_give %51 : memref<?x?x128xf16>
                  %102 = linalg.copy ins(%89 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %57 : memref<?x?x1xf16>
                  scf.yield %102, %95, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %45 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %73 = loom.broadcast ins(%72#1 : tensor<?x?x1xf16>) outs(%50 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %41 : memref<?x?x1xf16>
                %74 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %76 = loom.init_tensor %75[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72#2, %73 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%76 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %80 = arith.divf %in, %in_3 : f16
                  linalg.yield %80 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %49 : memref<?x?x128xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %78 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %79 = loom.bufferize_to_memref %77 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %79, %78 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %75 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x4x2_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc4_dim_y_level0_bc8_dim_x_level0_bc4_dim_y_level0_bc8_n(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
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
                %32 = arith.muli %arg4, %c4 : index
                %33 = arith.addi %arg5, %32 : index
                loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %34 = loom.bufferize_to_tensor %30[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %35 = arith.ceildivui %c4096, %20 : index
                %36 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %37 = loom.semaphore_take %36 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %40 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %41 = loom.semaphore_take %40 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %42 = loom.init_tensor %41[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %43 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %44 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %45 = loom.semaphore_take %44 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %46 = loom.init_tensor %45[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %48 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %49 = loom.semaphore_take %48 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %50 = loom.init_tensor %49[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %51 = loom.semaphore_take %48 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %52 = loom.init_tensor %51[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %53 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %54 = loom.semaphore_take %53 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %55 = loom.init_tensor %54[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %56 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %57 = loom.semaphore_take %56 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %58 = loom.init_tensor %57[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %59 = loom.alloc [%18, %19, 1] on @L1 : memref<?x?x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %61 = loom.init_tensor %60[%18, %19, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %62 = loom.alloc [%18, 128, %20] on @L1 : memref<?x128x?xf16>
                %63 = loom.semaphore_take %62 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %64 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %66 = loom.init_tensor %65[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %67 = loom.alloc [%18, %19, %20] on @L1 : memref<?x?x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %69 = loom.init_tensor %68[%18, %19, %20] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %70 = loom.alloc [%18, %20, 128] on @L1 : memref<?x?x128xf16>
                %71 = loom.semaphore_take %70 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %72:3 = scf.for %arg9 = %c0 to %35 step %c1 iter_args(%arg10 = %47, %arg11 = %43, %arg12 = %39) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %80 = arith.muli %arg9, %20 : index
                  %81 = loom.subview %arg0[%27, 0, %80] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  %82 = arith.addi %32, %c3 : index
                  loom.copy %81, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%32, %c0], LR : [%82, %c7]) : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %83 = loom.bufferize_to_tensor %63[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %84 = linalg.fill ins(%cst : f16) outs(%66 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %85 = linalg.batch_matmul ins(%34, %83 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%84 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %63 : memref<?x128x?xf16>
                  %86 = linalg.fill ins(%cst_1 : f16) outs(%58 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%85 : tensor<?x?x?xf16>) outs(%86 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.maximumf %in, %out : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87 : tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %88 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.cmpf ogt, %in, %in_3 : f16
                    %104 = arith.select %103, %in, %in_3 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x1xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85 : tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x?xf16>
                  %91 = loom.broadcast ins(%89 : tensor<?x?x1xf16>) outs(%69 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %91 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%66 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.subf %in, %in_3 : f16
                    %104 = math.exp %103 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x?x?xf16>
                  %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %89 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%61 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.subf %in, %in_3 : f16
                    %104 = math.exp %103 : f16
                    linalg.yield %104 : f16
                  } -> tensor<?x?x1xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %93 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.mulf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<?x?x?xf16>) outs(%94 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %103 = arith.addf %in, %out : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg12, %93 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%52 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.mulf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %60 : memref<?x?x1xf16>
                  %97 = loom.subview %arg1[%27, %80, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %97, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%32, %c0], LR : [%82, %c7]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %98 = loom.bufferize_to_tensor %71[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %99 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %100 = linalg.batch_matmul ins(%92, %98 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%99 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %71 : memref<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x?xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%100, %96 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %103 = arith.addf %in, %in_3 : f16
                    linalg.yield %103 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %54 : memref<?x?x128xf16>
                  loom.semaphore_give %51 : memref<?x?x128xf16>
                  %102 = linalg.copy ins(%89 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %57 : memref<?x?x1xf16>
                  scf.yield %102, %95, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %45 : memref<?x?x1xf16>
                loom.semaphore_give %30 : memref<?x?x128xf16>
                %73 = loom.broadcast ins(%72#1 : tensor<?x?x1xf16>) outs(%50 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %41 : memref<?x?x1xf16>
                %74 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %76 = loom.init_tensor %75[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%72#2, %73 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%76 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %80 = arith.divf %in, %in_3 : f16
                  linalg.yield %80 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %49 : memref<?x?x128xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %78 = loom.subview %arg3[%27, %28, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %79 = loom.bufferize_to_memref %77 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %79, %78 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %75 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8x1_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level1_bc8_dim_y_level0_bc8_dim_x_level1_bc8_dim_y_level0_bc8_n(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
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
                %30 = arith.muli %arg4, %c8 : index
                %31 = arith.addi %arg5, %30 : index
                loom.copy %29, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%31, %arg6], LR : [%31, %arg6]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %32 = loom.bufferize_to_tensor %28[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
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
                %51 = loom.alloc [%18, %19, 128] on @L1 : memref<?x?x128xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %53 = loom.init_tensor %52[%18, %19, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
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
                  %79 = loom.subview %arg0[%25, 0, %78] [%18, 128, %20] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %79, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %80 = loom.bufferize_to_tensor %61[%18, 128, %20] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %81 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %82 = linalg.batch_matmul ins(%32, %80 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%81 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %61 : memref<?x128x?xf16>
                  %83 = linalg.fill ins(%cst_1 : f16) outs(%56 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%82 : tensor<?x?x?xf16>) outs(%83 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %100 = arith.maximumf %in, %out : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84 : tensor<?x?x1xf16>) outs(%56 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %100 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %85 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%56 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.cmpf ogt, %in, %in_3 : f16
                    %101 = arith.select %100, %in, %in_3 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82 : tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %100 = arith.mulf %in, %cst_2 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x?xf16>
                  %88 = loom.broadcast ins(%86 : tensor<?x?x1xf16>) outs(%67 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
                  %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %88 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%64 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.subf %in, %in_3 : f16
                    %101 = math.exp %100 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x?x?xf16>
                  %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %86 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%59 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.subf %in, %in_3 : f16
                    %101 = math.exp %100 : f16
                    linalg.yield %101 : f16
                  } -> tensor<?x?x1xf16>
                  %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %90 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.mulf %in, %in_3 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%89 : tensor<?x?x?xf16>) outs(%91 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %100 = arith.addf %in, %out : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x1xf16>
                  %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg12, %90 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%50 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.mulf %in, %in_3 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %58 : memref<?x?x1xf16>
                  %94 = loom.subview %arg1[%25, %78, 0] [%18, %20, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = true] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %94, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %95 = loom.bufferize_to_tensor %69[%18, %20, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %96 = linalg.fill ins(%cst : f16) outs(%53 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %97 = linalg.batch_matmul ins(%89, %95 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%96 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %69 : memref<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x?xf16>
                  %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%97, %93 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %100 = arith.addf %in, %in_3 : f16
                    linalg.yield %100 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %52 : memref<?x?x128xf16>
                  loom.semaphore_give %49 : memref<?x?x128xf16>
                  %99 = linalg.copy ins(%86 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %55 : memref<?x?x1xf16>
                  scf.yield %99, %92, %98 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %43 : memref<?x?x1xf16>
                loom.semaphore_give %28 : memref<?x?x128xf16>
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
                %76 = loom.subview %arg3[%25, %26, 0] [%18, %19, 128] [1, 1, 1], reuse : [seq = false, spat = true, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %77 = loom.bufferize_to_memref %75 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%31, %arg6], LR : [%31, %arg6]) : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %73 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
