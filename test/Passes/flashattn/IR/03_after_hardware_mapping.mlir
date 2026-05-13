module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
      %20 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c32, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (1) {
            %25 = arith.ceildivui %23, %c8 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c8 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %35 = loom.init_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %36 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %37 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %39, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %40 = loom.bufferize_to_tensor %36[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %41 = loom.sync ins(%40 : tensor<?x?x128xf16>) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %36 : memref<?x?x128xf16>
                %42 = arith.ceildivui %c4096, %22 : index
                %43 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %44 = loom.semaphore_take %43 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %45 = loom.init_tensor %44[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %46 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %47 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %49 = loom.init_tensor %48[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %51 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %55 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %57 = loom.init_tensor %56[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %58 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %60 = loom.init_tensor %59[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %61 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %63 = loom.init_tensor %62[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %64 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %66 = loom.init_tensor %65[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.init_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %70 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %71 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                %72 = loom.semaphore_take %71 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %73 = loom.init_tensor %72[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %74 = loom.alloc [%20, %21, 32] on @L1 : memref<?x?x32xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %76 = loom.init_tensor %75[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %77 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %78 = loom.init_tensor %77[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81 = loom.init_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %82 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %83:3 = scf.for %arg9 = %c0 to %42 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %46) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %89 = arith.muli %arg9, %22 : index
                  %90 = loom.subview %arg0[%29, 0, %89] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %90, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %91 = loom.bufferize_to_tensor %70[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %92 = loom.sync ins(%91 : tensor<?x128x?xf16>) outs(%69 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
                  loom.semaphore_give %70 : memref<?x128x?xf16>
                  %93 = linalg.fill ins(%cst : f16) outs(%73 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %94 = linalg.batch_matmul ins(%41, %92 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%93 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  %95 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%94 : tensor<?x?x?xf16>) outs(%95 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.maximumf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%60 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in_3, %cst_2 : f16
                    %112 = arith.cmpf ogt, %in, %111 : f16
                    %113 = arith.select %112, %in, %111 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x1xf16>
                  %98 = loom.broadcast ins(%97 : tensor<?x?x1xf16>) outs(%78 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94, %98 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%73 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in, %cst_2 : f16
                    %112 = arith.subf %111, %in_3 : f16
                    %113 = math.exp %112 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %77 : memref<?x?x32xf16>
                  %100 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%99 : tensor<?x?x?xf16>) outs(%100 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.addf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %97 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%66 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.subf %in, %in_3 : f16
                    %112 = math.exp %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %102, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in, %in_3 : f16
                    %112 = arith.addf %111, %in_4 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %62 : memref<?x?x1xf16>
                  %104 = loom.subview %arg1[%29, %89, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %104, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %105 = loom.bufferize_to_tensor %82[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %106 = loom.sync ins(%105 : tensor<?x?x128xf16>) outs(%81 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %82 : memref<?x?x128xf16>
                  %107 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %108 = linalg.batch_matmul ins(%99, %106 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%107 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %72 : memref<?x?x?xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%108, %arg12, %102 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in_3, %in_4 : f16
                    %112 = arith.addf %in, %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x1xf16>
                  loom.semaphore_give %56 : memref<?x?x128xf16>
                  %110 = linalg.copy ins(%97 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %59 : memref<?x?x1xf16>
                  scf.yield %110, %103, %109 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %52 : memref<?x?x1xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %84 = loom.broadcast ins(%83#1 : tensor<?x?x1xf16>) outs(%76 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %48 : memref<?x?x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83#2, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%35 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %89 = arith.divf %in, %in_3 : f16
                  linalg.yield %89 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %75 : memref<?x?x32xf16>
                loom.semaphore_give %44 : memref<?x?x128xf16>
                %86 = loom.sync ins(%85 : tensor<?x?x128xf16>) outs(%33 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %34 : memref<?x?x128xf16>
                %87 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %88 = loom.bufferize_to_memref %86 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %88, %87 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %32 : memref<?x?x128xf16>
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
      %20 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c32, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (2) {
            %25 = arith.ceildivui %23, %c4 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c16 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg5, %arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %35 = loom.init_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %36 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %37 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %39, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %40 = loom.bufferize_to_tensor %36[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %41 = loom.sync ins(%40 : tensor<?x?x128xf16>) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %36 : memref<?x?x128xf16>
                %42 = arith.ceildivui %c4096, %22 : index
                %43 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %44 = loom.semaphore_take %43 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %45 = loom.init_tensor %44[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %46 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %47 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %49 = loom.init_tensor %48[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %51 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %55 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %57 = loom.init_tensor %56[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %58 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %60 = loom.init_tensor %59[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %61 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %63 = loom.init_tensor %62[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %64 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %66 = loom.init_tensor %65[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.init_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %70 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %71 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                %72 = loom.semaphore_take %71 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %73 = loom.init_tensor %72[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %74 = loom.alloc [%20, %21, 32] on @L1 : memref<?x?x32xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %76 = loom.init_tensor %75[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %77 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %78 = loom.init_tensor %77[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81 = loom.init_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %82 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %83:3 = scf.for %arg9 = %c0 to %42 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %46) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %89 = arith.muli %arg9, %22 : index
                  %90 = loom.subview %arg0[%29, 0, %89] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %90, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %91 = loom.bufferize_to_tensor %70[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %92 = loom.sync ins(%91 : tensor<?x128x?xf16>) outs(%69 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
                  loom.semaphore_give %70 : memref<?x128x?xf16>
                  %93 = linalg.fill ins(%cst : f16) outs(%73 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %94 = linalg.batch_matmul ins(%41, %92 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%93 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  %95 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%94 : tensor<?x?x?xf16>) outs(%95 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.maximumf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%60 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in_3, %cst_2 : f16
                    %112 = arith.cmpf ogt, %in, %111 : f16
                    %113 = arith.select %112, %in, %111 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x1xf16>
                  %98 = loom.broadcast ins(%97 : tensor<?x?x1xf16>) outs(%78 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94, %98 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%73 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in, %cst_2 : f16
                    %112 = arith.subf %111, %in_3 : f16
                    %113 = math.exp %112 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %77 : memref<?x?x32xf16>
                  %100 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%99 : tensor<?x?x?xf16>) outs(%100 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.addf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %97 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%66 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.subf %in, %in_3 : f16
                    %112 = math.exp %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %102, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in, %in_3 : f16
                    %112 = arith.addf %111, %in_4 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %62 : memref<?x?x1xf16>
                  %104 = loom.subview %arg1[%29, %89, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %104, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %105 = loom.bufferize_to_tensor %82[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %106 = loom.sync ins(%105 : tensor<?x?x128xf16>) outs(%81 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %82 : memref<?x?x128xf16>
                  %107 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %108 = linalg.batch_matmul ins(%99, %106 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%107 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %72 : memref<?x?x?xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%108, %arg12, %102 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in_3, %in_4 : f16
                    %112 = arith.addf %in, %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x1xf16>
                  loom.semaphore_give %56 : memref<?x?x128xf16>
                  %110 = linalg.copy ins(%97 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %59 : memref<?x?x1xf16>
                  scf.yield %110, %103, %109 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %52 : memref<?x?x1xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %84 = loom.broadcast ins(%83#1 : tensor<?x?x1xf16>) outs(%76 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %48 : memref<?x?x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83#2, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%35 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %89 = arith.divf %in, %in_3 : f16
                  linalg.yield %89 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %75 : memref<?x?x32xf16>
                loom.semaphore_give %44 : memref<?x?x128xf16>
                %86 = loom.sync ins(%85 : tensor<?x?x128xf16>) outs(%33 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %34 : memref<?x?x128xf16>
                %87 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %88 = loom.bufferize_to_memref %86 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %88, %87 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %32 : memref<?x?x128xf16>
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
      %20 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c32, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (4) {
            %25 = arith.ceildivui %23, %c2 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c32 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg5, %arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %35 = loom.init_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %36 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %37 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %39, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %40 = loom.bufferize_to_tensor %36[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %41 = loom.sync ins(%40 : tensor<?x?x128xf16>) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %36 : memref<?x?x128xf16>
                %42 = arith.ceildivui %c4096, %22 : index
                %43 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %44 = loom.semaphore_take %43 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %45 = loom.init_tensor %44[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %46 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %47 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %49 = loom.init_tensor %48[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %51 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %55 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %57 = loom.init_tensor %56[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %58 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %60 = loom.init_tensor %59[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %61 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %63 = loom.init_tensor %62[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %64 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %66 = loom.init_tensor %65[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.init_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %70 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %71 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                %72 = loom.semaphore_take %71 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %73 = loom.init_tensor %72[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %74 = loom.alloc [%20, %21, 32] on @L1 : memref<?x?x32xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %76 = loom.init_tensor %75[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %77 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %78 = loom.init_tensor %77[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81 = loom.init_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %82 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %83:3 = scf.for %arg9 = %c0 to %42 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %46) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %89 = arith.muli %arg9, %22 : index
                  %90 = loom.subview %arg0[%29, 0, %89] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %90, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %91 = loom.bufferize_to_tensor %70[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %92 = loom.sync ins(%91 : tensor<?x128x?xf16>) outs(%69 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
                  loom.semaphore_give %70 : memref<?x128x?xf16>
                  %93 = linalg.fill ins(%cst : f16) outs(%73 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %94 = linalg.batch_matmul ins(%41, %92 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%93 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  %95 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%94 : tensor<?x?x?xf16>) outs(%95 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.maximumf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%60 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in_3, %cst_2 : f16
                    %112 = arith.cmpf ogt, %in, %111 : f16
                    %113 = arith.select %112, %in, %111 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x1xf16>
                  %98 = loom.broadcast ins(%97 : tensor<?x?x1xf16>) outs(%78 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94, %98 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%73 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in, %cst_2 : f16
                    %112 = arith.subf %111, %in_3 : f16
                    %113 = math.exp %112 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %77 : memref<?x?x32xf16>
                  %100 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%99 : tensor<?x?x?xf16>) outs(%100 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.addf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %97 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%66 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.subf %in, %in_3 : f16
                    %112 = math.exp %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %102, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in, %in_3 : f16
                    %112 = arith.addf %111, %in_4 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %62 : memref<?x?x1xf16>
                  %104 = loom.subview %arg1[%29, %89, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %104, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %105 = loom.bufferize_to_tensor %82[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %106 = loom.sync ins(%105 : tensor<?x?x128xf16>) outs(%81 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %82 : memref<?x?x128xf16>
                  %107 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %108 = linalg.batch_matmul ins(%99, %106 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%107 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %72 : memref<?x?x?xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%108, %arg12, %102 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in_3, %in_4 : f16
                    %112 = arith.addf %in, %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x1xf16>
                  loom.semaphore_give %56 : memref<?x?x128xf16>
                  %110 = linalg.copy ins(%97 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %59 : memref<?x?x1xf16>
                  scf.yield %110, %103, %109 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %52 : memref<?x?x1xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %84 = loom.broadcast ins(%83#1 : tensor<?x?x1xf16>) outs(%76 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %48 : memref<?x?x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83#2, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%35 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %89 = arith.divf %in, %in_3 : f16
                  linalg.yield %89 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %75 : memref<?x?x32xf16>
                loom.semaphore_give %44 : memref<?x?x128xf16>
                %86 = loom.sync ins(%85 : tensor<?x?x128xf16>) outs(%33 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %34 : memref<?x?x128xf16>
                %87 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %88 = loom.bufferize_to_memref %86 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %88, %87 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %32 : memref<?x?x128xf16>
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
      %20 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c32, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %23 step %c1 {
              %25 = arith.ceildivui %24, %c64 : index
              scf.for %arg8 = %c0 to %25 step %c1 {
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %27 = arith.muli %arg7, %20 : index
                %28 = arith.muli %26, %21 : index
                %29 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %31 = loom.init_tensor %30[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %32 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %35 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %36 = loom.init_tensor %35[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %37 = loom.subview %arg2[%27, %28, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %37, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %38 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = loom.sync ins(%38 : tensor<?x?x128xf16>) outs(%36 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %34 : memref<?x?x128xf16>
                %40 = arith.ceildivui %c4096, %22 : index
                %41 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %42 = loom.semaphore_take %41 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %43 = loom.init_tensor %42[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %44 = linalg.fill ins(%cst : f16) outs(%43 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %45 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %46 = loom.semaphore_take %45 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %47 = loom.init_tensor %46[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %48 = linalg.fill ins(%cst_0 : f16) outs(%47 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %49 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %50 = loom.semaphore_take %49 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %51 = loom.init_tensor %50[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %52 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %53 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %54 = loom.semaphore_take %53 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %55 = loom.init_tensor %54[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %56 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %57 = loom.semaphore_take %56 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %58 = loom.init_tensor %57[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %59 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %61 = loom.init_tensor %60[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %62 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %63 = loom.semaphore_take %62 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %64 = loom.init_tensor %63[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %65 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %67 = loom.init_tensor %66[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %68 = loom.semaphore_take %65 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                %70 = loom.semaphore_take %69 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %71 = loom.init_tensor %70[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %72 = loom.alloc [%20, %21, 32] on @L1 : memref<?x?x32xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %74 = loom.init_tensor %73[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %75 = loom.semaphore_take %72 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %76 = loom.init_tensor %75[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %77 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %78 = loom.semaphore_take %77 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %79 = loom.init_tensor %78[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %80 = loom.semaphore_take %77 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81:3 = scf.for %arg9 = %c0 to %40 step %c1 iter_args(%arg10 = %52, %arg11 = %48, %arg12 = %44) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %87 = arith.muli %arg9, %22 : index
                  %88 = loom.subview %arg0[%27, 0, %87] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %88, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %89 = loom.bufferize_to_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %90 = loom.sync ins(%89 : tensor<?x128x?xf16>) outs(%67 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  %91 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %92 = linalg.batch_matmul ins(%39, %90 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%91 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x128x?xf16>
                  %93 = linalg.fill ins(%cst_1 : f16) outs(%58 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<?x?x?xf16>) outs(%93 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %109 = arith.maximumf %in, %out : f16
                    linalg.yield %109 : f16
                  } -> tensor<?x?x1xf16>
                  %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %94 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %109 = arith.mulf %in_3, %cst_2 : f16
                    %110 = arith.cmpf ogt, %in, %109 : f16
                    %111 = arith.select %110, %in, %109 : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %96 = loom.broadcast ins(%95 : tensor<?x?x1xf16>) outs(%76 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%92, %96 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%71 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %109 = arith.mulf %in, %cst_2 : f16
                    %110 = arith.subf %109, %in_3 : f16
                    %111 = math.exp %110 : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %75 : memref<?x?x32xf16>
                  %98 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%97 : tensor<?x?x?xf16>) outs(%98 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %109 = arith.addf %in, %out : f16
                    linalg.yield %109 : f16
                  } -> tensor<?x?x1xf16>
                  %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %95 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%64 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %109 = arith.subf %in, %in_3 : f16
                    %110 = math.exp %109 : f16
                    linalg.yield %110 : f16
                  } -> tensor<?x?x1xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %100, %99 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %109 = arith.mulf %in, %in_3 : f16
                    %110 = arith.addf %109, %in_4 : f16
                    linalg.yield %110 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %60 : memref<?x?x1xf16>
                  %102 = loom.subview %arg1[%27, %87, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %102, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %103 = loom.bufferize_to_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %104 = loom.sync ins(%103 : tensor<?x?x128xf16>) outs(%79 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  %105 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %106 = linalg.batch_matmul ins(%97, %104 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%105 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %78 : memref<?x?x128xf16>
                  loom.semaphore_give %70 : memref<?x?x?xf16>
                  %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %arg12, %100 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %109 = arith.mulf %in_3, %in_4 : f16
                    %110 = arith.addf %in, %109 : f16
                    linalg.yield %110 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x1xf16>
                  loom.semaphore_give %54 : memref<?x?x128xf16>
                  %108 = linalg.copy ins(%95 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %57 : memref<?x?x1xf16>
                  scf.yield %108, %101, %107 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %50 : memref<?x?x1xf16>
                loom.semaphore_give %35 : memref<?x?x128xf16>
                %82 = loom.broadcast ins(%81#1 : tensor<?x?x1xf16>) outs(%74 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %46 : memref<?x?x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#2, %82 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%33 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %87 = arith.divf %in, %in_3 : f16
                  linalg.yield %87 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %73 : memref<?x?x32xf16>
                loom.semaphore_give %42 : memref<?x?x128xf16>
                %84 = loom.sync ins(%83 : tensor<?x?x128xf16>) outs(%31 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %32 : memref<?x?x128xf16>
                %85 = loom.subview %arg3[%27, %28, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %86 = loom.bufferize_to_memref %84 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %86, %85 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %30 : memref<?x?x128xf16>
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
      %20 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c32, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (1) {
          affine.parallel (%arg6) = (0) to (8) {
            %25 = arith.ceildivui %23, %c8 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c8 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %35 = loom.init_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %36 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %37 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %39, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %40 = loom.bufferize_to_tensor %36[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %41 = loom.sync ins(%40 : tensor<?x?x128xf16>) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %36 : memref<?x?x128xf16>
                %42 = arith.ceildivui %c4096, %22 : index
                %43 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %44 = loom.semaphore_take %43 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %45 = loom.init_tensor %44[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %46 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %47 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %49 = loom.init_tensor %48[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %51 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %55 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %57 = loom.init_tensor %56[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %58 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %60 = loom.init_tensor %59[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %61 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %63 = loom.init_tensor %62[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %64 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %66 = loom.init_tensor %65[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.init_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %70 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %71 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                %72 = loom.semaphore_take %71 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %73 = loom.init_tensor %72[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %74 = loom.alloc [%20, %21, 32] on @L1 : memref<?x?x32xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %76 = loom.init_tensor %75[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %77 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %78 = loom.init_tensor %77[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81 = loom.init_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %82 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %83:3 = scf.for %arg9 = %c0 to %42 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %46) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %89 = arith.muli %arg9, %22 : index
                  %90 = loom.subview %arg0[%29, 0, %89] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %90, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %91 = loom.bufferize_to_tensor %70[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %92 = loom.sync ins(%91 : tensor<?x128x?xf16>) outs(%69 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
                  loom.semaphore_give %70 : memref<?x128x?xf16>
                  %93 = linalg.fill ins(%cst : f16) outs(%73 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %94 = linalg.batch_matmul ins(%41, %92 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%93 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  %95 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%94 : tensor<?x?x?xf16>) outs(%95 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.maximumf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%60 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in_3, %cst_2 : f16
                    %112 = arith.cmpf ogt, %in, %111 : f16
                    %113 = arith.select %112, %in, %111 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x1xf16>
                  %98 = loom.broadcast ins(%97 : tensor<?x?x1xf16>) outs(%78 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94, %98 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%73 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in, %cst_2 : f16
                    %112 = arith.subf %111, %in_3 : f16
                    %113 = math.exp %112 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %77 : memref<?x?x32xf16>
                  %100 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%99 : tensor<?x?x?xf16>) outs(%100 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.addf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %97 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%66 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.subf %in, %in_3 : f16
                    %112 = math.exp %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %102, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in, %in_3 : f16
                    %112 = arith.addf %111, %in_4 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %62 : memref<?x?x1xf16>
                  %104 = loom.subview %arg1[%29, %89, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %104, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %105 = loom.bufferize_to_tensor %82[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %106 = loom.sync ins(%105 : tensor<?x?x128xf16>) outs(%81 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %82 : memref<?x?x128xf16>
                  %107 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %108 = linalg.batch_matmul ins(%99, %106 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%107 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %72 : memref<?x?x?xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%108, %arg12, %102 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in_3, %in_4 : f16
                    %112 = arith.addf %in, %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x1xf16>
                  loom.semaphore_give %56 : memref<?x?x128xf16>
                  %110 = linalg.copy ins(%97 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %59 : memref<?x?x1xf16>
                  scf.yield %110, %103, %109 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %52 : memref<?x?x1xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %84 = loom.broadcast ins(%83#1 : tensor<?x?x1xf16>) outs(%76 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %48 : memref<?x?x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83#2, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%35 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %89 = arith.divf %in, %in_3 : f16
                  linalg.yield %89 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %75 : memref<?x?x32xf16>
                loom.semaphore_give %44 : memref<?x?x128xf16>
                %86 = loom.sync ins(%85 : tensor<?x?x128xf16>) outs(%33 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %34 : memref<?x?x128xf16>
                %87 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %88 = loom.bufferize_to_memref %86 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %88, %87 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %32 : memref<?x?x128xf16>
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
      %20 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c32, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (2) {
          affine.parallel (%arg6) = (0) to (8) {
            %25 = arith.ceildivui %23, %c4 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c16 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg5, %arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %35 = loom.init_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %36 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %37 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %39, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %40 = loom.bufferize_to_tensor %36[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %41 = loom.sync ins(%40 : tensor<?x?x128xf16>) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %36 : memref<?x?x128xf16>
                %42 = arith.ceildivui %c4096, %22 : index
                %43 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %44 = loom.semaphore_take %43 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %45 = loom.init_tensor %44[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %46 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %47 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %49 = loom.init_tensor %48[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %51 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %55 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %57 = loom.init_tensor %56[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %58 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %60 = loom.init_tensor %59[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %61 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %63 = loom.init_tensor %62[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %64 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %66 = loom.init_tensor %65[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.init_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %70 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %71 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                %72 = loom.semaphore_take %71 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %73 = loom.init_tensor %72[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %74 = loom.alloc [%20, %21, 32] on @L1 : memref<?x?x32xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %76 = loom.init_tensor %75[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %77 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %78 = loom.init_tensor %77[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81 = loom.init_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %82 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %83:3 = scf.for %arg9 = %c0 to %42 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %46) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %89 = arith.muli %arg9, %22 : index
                  %90 = loom.subview %arg0[%29, 0, %89] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %90, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %91 = loom.bufferize_to_tensor %70[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %92 = loom.sync ins(%91 : tensor<?x128x?xf16>) outs(%69 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
                  loom.semaphore_give %70 : memref<?x128x?xf16>
                  %93 = linalg.fill ins(%cst : f16) outs(%73 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %94 = linalg.batch_matmul ins(%41, %92 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%93 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  %95 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%94 : tensor<?x?x?xf16>) outs(%95 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.maximumf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%60 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in_3, %cst_2 : f16
                    %112 = arith.cmpf ogt, %in, %111 : f16
                    %113 = arith.select %112, %in, %111 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x1xf16>
                  %98 = loom.broadcast ins(%97 : tensor<?x?x1xf16>) outs(%78 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94, %98 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%73 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in, %cst_2 : f16
                    %112 = arith.subf %111, %in_3 : f16
                    %113 = math.exp %112 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %77 : memref<?x?x32xf16>
                  %100 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%99 : tensor<?x?x?xf16>) outs(%100 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.addf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %97 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%66 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.subf %in, %in_3 : f16
                    %112 = math.exp %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %102, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in, %in_3 : f16
                    %112 = arith.addf %111, %in_4 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %62 : memref<?x?x1xf16>
                  %104 = loom.subview %arg1[%29, %89, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %104, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %105 = loom.bufferize_to_tensor %82[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %106 = loom.sync ins(%105 : tensor<?x?x128xf16>) outs(%81 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %82 : memref<?x?x128xf16>
                  %107 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %108 = linalg.batch_matmul ins(%99, %106 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%107 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %72 : memref<?x?x?xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%108, %arg12, %102 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in_3, %in_4 : f16
                    %112 = arith.addf %in, %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x1xf16>
                  loom.semaphore_give %56 : memref<?x?x128xf16>
                  %110 = linalg.copy ins(%97 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %59 : memref<?x?x1xf16>
                  scf.yield %110, %103, %109 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %52 : memref<?x?x1xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %84 = loom.broadcast ins(%83#1 : tensor<?x?x1xf16>) outs(%76 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %48 : memref<?x?x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83#2, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%35 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %89 = arith.divf %in, %in_3 : f16
                  linalg.yield %89 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %75 : memref<?x?x32xf16>
                loom.semaphore_give %44 : memref<?x?x128xf16>
                %86 = loom.sync ins(%85 : tensor<?x?x128xf16>) outs(%33 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %34 : memref<?x?x128xf16>
                %87 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %88 = loom.bufferize_to_memref %86 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %88, %87 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %32 : memref<?x?x128xf16>
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
      %20 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c32, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (4) {
          affine.parallel (%arg6) = (0) to (8) {
            %25 = arith.ceildivui %23, %c2 : index
            scf.for %arg7 = %c0 to %25 step %c1 {
              %26 = arith.ceildivui %24, %c32 : index
              scf.for %arg8 = %c0 to %26 step %c1 {
                %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
                %28 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg5, %arg6, %arg8)
                %29 = arith.muli %27, %20 : index
                %30 = arith.muli %28, %21 : index
                %31 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %35 = loom.init_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %36 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %37 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %38 = loom.init_tensor %37[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = loom.subview %arg2[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %39, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %40 = loom.bufferize_to_tensor %36[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %41 = loom.sync ins(%40 : tensor<?x?x128xf16>) outs(%38 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %36 : memref<?x?x128xf16>
                %42 = arith.ceildivui %c4096, %22 : index
                %43 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %44 = loom.semaphore_take %43 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %45 = loom.init_tensor %44[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %46 = linalg.fill ins(%cst : f16) outs(%45 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %47 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %48 = loom.semaphore_take %47 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %49 = loom.init_tensor %48[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %50 = linalg.fill ins(%cst_0 : f16) outs(%49 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %51 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %52 = loom.semaphore_take %51 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %53 = loom.init_tensor %52[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %54 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %55 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %56 = loom.semaphore_take %55 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %57 = loom.init_tensor %56[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %58 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %59 = loom.semaphore_take %58 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %60 = loom.init_tensor %59[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %61 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %62 = loom.semaphore_take %61 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %63 = loom.init_tensor %62[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %64 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %65 = loom.semaphore_take %64 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %66 = loom.init_tensor %65[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %67 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %68 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.init_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %70 = loom.semaphore_take %67 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %71 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                %72 = loom.semaphore_take %71 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %73 = loom.init_tensor %72[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %74 = loom.alloc [%20, %21, 32] on @L1 : memref<?x?x32xf16>
                %75 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %76 = loom.init_tensor %75[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %77 = loom.semaphore_take %74 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %78 = loom.init_tensor %77[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %79 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %80 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81 = loom.init_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %82 = loom.semaphore_take %79 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %83:3 = scf.for %arg9 = %c0 to %42 step %c1 iter_args(%arg10 = %54, %arg11 = %50, %arg12 = %46) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %89 = arith.muli %arg9, %22 : index
                  %90 = loom.subview %arg0[%29, 0, %89] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %90, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %91 = loom.bufferize_to_tensor %70[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %92 = loom.sync ins(%91 : tensor<?x128x?xf16>) outs(%69 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
                  loom.semaphore_give %70 : memref<?x128x?xf16>
                  %93 = linalg.fill ins(%cst : f16) outs(%73 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %94 = linalg.batch_matmul ins(%41, %92 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%93 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  %95 = linalg.fill ins(%cst_1 : f16) outs(%60 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%94 : tensor<?x?x?xf16>) outs(%95 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.maximumf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %96 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%60 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in_3, %cst_2 : f16
                    %112 = arith.cmpf ogt, %in, %111 : f16
                    %113 = arith.select %112, %in, %111 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x1xf16>
                  %98 = loom.broadcast ins(%97 : tensor<?x?x1xf16>) outs(%78 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%94, %98 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%73 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.mulf %in, %cst_2 : f16
                    %112 = arith.subf %111, %in_3 : f16
                    %113 = math.exp %112 : f16
                    linalg.yield %113 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %77 : memref<?x?x32xf16>
                  %100 = linalg.fill ins(%cst : f16) outs(%63 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%99 : tensor<?x?x?xf16>) outs(%100 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %111 = arith.addf %in, %out : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %97 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%66 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %111 = arith.subf %in, %in_3 : f16
                    %112 = math.exp %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %102, %101 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in, %in_3 : f16
                    %112 = arith.addf %111, %in_4 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %62 : memref<?x?x1xf16>
                  %104 = loom.subview %arg1[%29, %89, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %104, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %105 = loom.bufferize_to_tensor %82[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %106 = loom.sync ins(%105 : tensor<?x?x128xf16>) outs(%81 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %82 : memref<?x?x128xf16>
                  %107 = linalg.fill ins(%cst : f16) outs(%57 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %108 = linalg.batch_matmul ins(%99, %106 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%107 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  loom.semaphore_give %72 : memref<?x?x?xf16>
                  %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%108, %arg12, %102 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %111 = arith.mulf %in_3, %in_4 : f16
                    %112 = arith.addf %in, %111 : f16
                    linalg.yield %112 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %65 : memref<?x?x1xf16>
                  loom.semaphore_give %56 : memref<?x?x128xf16>
                  %110 = linalg.copy ins(%97 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %59 : memref<?x?x1xf16>
                  scf.yield %110, %103, %109 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %52 : memref<?x?x1xf16>
                loom.semaphore_give %37 : memref<?x?x128xf16>
                %84 = loom.broadcast ins(%83#1 : tensor<?x?x1xf16>) outs(%76 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %48 : memref<?x?x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%83#2, %84 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%35 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %89 = arith.divf %in, %in_3 : f16
                  linalg.yield %89 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %75 : memref<?x?x32xf16>
                loom.semaphore_give %44 : memref<?x?x128xf16>
                %86 = loom.sync ins(%85 : tensor<?x?x128xf16>) outs(%33 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %34 : memref<?x?x128xf16>
                %87 = loom.subview %arg3[%29, %30, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %88 = loom.bufferize_to_memref %86 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %88, %87 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %32 : memref<?x?x128xf16>
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
      %20 = loom.sym @tile_b {upper_bound = 32 : index} : index
      %21 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %23 = arith.ceildivui %c32, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %23 step %c1 {
              %25 = arith.ceildivui %24, %c64 : index
              scf.for %arg8 = %c0 to %25 step %c1 {
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg5, %arg6, %arg8)
                %27 = arith.muli %arg7, %20 : index
                %28 = arith.muli %26, %21 : index
                %29 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %30 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %31 = loom.init_tensor %30[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %32 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %33 = loom.init_tensor %32[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %34 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %35 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %36 = loom.init_tensor %35[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %37 = loom.subview %arg2[%27, %28, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %37, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                %38 = loom.bufferize_to_tensor %34[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %39 = loom.sync ins(%38 : tensor<?x?x128xf16>) outs(%36 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %34 : memref<?x?x128xf16>
                %40 = arith.ceildivui %c4096, %22 : index
                %41 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %42 = loom.semaphore_take %41 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %43 = loom.init_tensor %42[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %44 = linalg.fill ins(%cst : f16) outs(%43 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                %45 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %46 = loom.semaphore_take %45 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %47 = loom.init_tensor %46[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %48 = linalg.fill ins(%cst_0 : f16) outs(%47 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %49 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %50 = loom.semaphore_take %49 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %51 = loom.init_tensor %50[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %52 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                %53 = loom.alloc [%20, %21, 128] on @L1 : memref<?x?x128xf16>
                %54 = loom.semaphore_take %53 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %55 = loom.init_tensor %54[%20, %21, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %56 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %57 = loom.semaphore_take %56 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %58 = loom.init_tensor %57[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %59 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %60 = loom.semaphore_take %59 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %61 = loom.init_tensor %60[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %62 = loom.alloc [%20, %21, 1] on @L1 : memref<?x?x1xf16>
                %63 = loom.semaphore_take %62 : memref<?x?x1xf16> -> memref<?x?x1xf16>
                %64 = loom.init_tensor %63[%20, %21, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
                %65 = loom.alloc [%20, 128, %22] on @L1 : memref<?x128x?xf16>
                %66 = loom.semaphore_take %65 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %67 = loom.init_tensor %66[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                %68 = loom.semaphore_take %65 : memref<?x128x?xf16> -> memref<?x128x?xf16>
                %69 = loom.alloc [%20, %21, %22] on @L1 : memref<?x?x?xf16>
                %70 = loom.semaphore_take %69 : memref<?x?x?xf16> -> memref<?x?x?xf16>
                %71 = loom.init_tensor %70[%20, %21, %22] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
                %72 = loom.alloc [%20, %21, 32] on @L1 : memref<?x?x32xf16>
                %73 = loom.semaphore_take %72 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %74 = loom.init_tensor %73[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %75 = loom.semaphore_take %72 : memref<?x?x32xf16> -> memref<?x?x32xf16>
                %76 = loom.init_tensor %75[%20, %21, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
                %77 = loom.alloc [%20, %22, 128] on @L1 : memref<?x?x128xf16>
                %78 = loom.semaphore_take %77 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %79 = loom.init_tensor %78[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                %80 = loom.semaphore_take %77 : memref<?x?x128xf16> -> memref<?x?x128xf16>
                %81:3 = scf.for %arg9 = %c0 to %40 step %c1 iter_args(%arg10 = %52, %arg11 = %48, %arg12 = %44) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
                  %87 = arith.muli %arg9, %22 : index
                  %88 = loom.subview %arg0[%27, 0, %87] [%20, 128, %22] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
                  loom.copy %88, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
                  %89 = loom.bufferize_to_tensor %68[%20, 128, %22] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
                  %90 = loom.sync ins(%89 : tensor<?x128x?xf16>) outs(%67 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
                  loom.semaphore_give %68 : memref<?x128x?xf16>
                  %91 = linalg.fill ins(%cst : f16) outs(%71 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  %92 = linalg.batch_matmul ins(%39, %90 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%91 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
                  loom.semaphore_give %66 : memref<?x128x?xf16>
                  %93 = linalg.fill ins(%cst_1 : f16) outs(%58 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<?x?x?xf16>) outs(%93 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %109 = arith.maximumf %in, %out : f16
                    linalg.yield %109 : f16
                  } -> tensor<?x?x1xf16>
                  %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %94 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%58 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %109 = arith.mulf %in_3, %cst_2 : f16
                    %110 = arith.cmpf ogt, %in, %109 : f16
                    %111 = arith.select %110, %in, %109 : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x1xf16>
                  %96 = loom.broadcast ins(%95 : tensor<?x?x1xf16>) outs(%76 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
                  %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%92, %96 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%71 : tensor<?x?x?xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %109 = arith.mulf %in, %cst_2 : f16
                    %110 = arith.subf %109, %in_3 : f16
                    %111 = math.exp %110 : f16
                    linalg.yield %111 : f16
                  } -> tensor<?x?x?xf16>
                  loom.semaphore_give %75 : memref<?x?x32xf16>
                  %98 = linalg.fill ins(%cst : f16) outs(%61 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%97 : tensor<?x?x?xf16>) outs(%98 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %109 = arith.addf %in, %out : f16
                    linalg.yield %109 : f16
                  } -> tensor<?x?x1xf16>
                  %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %95 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%64 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %out: f16):
                    %109 = arith.subf %in, %in_3 : f16
                    %110 = math.exp %109 : f16
                    linalg.yield %110 : f16
                  } -> tensor<?x?x1xf16>
                  %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %100, %99 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg11 : tensor<?x?x1xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %109 = arith.mulf %in, %in_3 : f16
                    %110 = arith.addf %109, %in_4 : f16
                    linalg.yield %110 : f16
                  } -> tensor<?x?x1xf16>
                  loom.semaphore_give %60 : memref<?x?x1xf16>
                  %102 = loom.subview %arg1[%27, %87, 0] [%20, %22, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                  loom.copy %102, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
                  %103 = loom.bufferize_to_tensor %80[%20, %22, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
                  %104 = loom.sync ins(%103 : tensor<?x?x128xf16>) outs(%79 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %80 : memref<?x?x128xf16>
                  %105 = linalg.fill ins(%cst : f16) outs(%55 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  %106 = linalg.batch_matmul ins(%97, %104 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%105 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                  loom.semaphore_give %78 : memref<?x?x128xf16>
                  loom.semaphore_give %70 : memref<?x?x?xf16>
                  %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %arg12, %100 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg12 : tensor<?x?x128xf16>) {
                  ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
                    %109 = arith.mulf %in_3, %in_4 : f16
                    %110 = arith.addf %in, %109 : f16
                    linalg.yield %110 : f16
                  } -> tensor<?x?x128xf16>
                  loom.semaphore_give %63 : memref<?x?x1xf16>
                  loom.semaphore_give %54 : memref<?x?x128xf16>
                  %108 = linalg.copy ins(%95 : tensor<?x?x1xf16>) outs(%arg10 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
                  loom.semaphore_give %57 : memref<?x?x1xf16>
                  scf.yield %108, %101, %107 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %50 : memref<?x?x1xf16>
                loom.semaphore_give %35 : memref<?x?x128xf16>
                %82 = loom.broadcast ins(%81#1 : tensor<?x?x1xf16>) outs(%74 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
                loom.semaphore_give %46 : memref<?x?x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81#2, %82 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%33 : tensor<?x?x128xf16>) {
                ^bb0(%in: f16, %in_3: f16, %out: f16):
                  %87 = arith.divf %in, %in_3 : f16
                  linalg.yield %87 : f16
                } -> tensor<?x?x128xf16>
                loom.semaphore_give %73 : memref<?x?x32xf16>
                loom.semaphore_give %42 : memref<?x?x128xf16>
                %84 = loom.sync ins(%83 : tensor<?x?x128xf16>) outs(%31 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
                loom.semaphore_give %32 : memref<?x?x128xf16>
                %85 = loom.subview %arg3[%27, %28, 0] [%20, %21, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                %86 = loom.bufferize_to_memref %84 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
                loom.copy %86, %85 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.semaphore_give %30 : memref<?x?x128xf16>
              } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
