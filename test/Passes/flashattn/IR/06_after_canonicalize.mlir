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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y1y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_x_level0_bc8_n__tile_b1__tile_m512__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
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
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %19 = arith.muli %arg5, %c512 : index
              %20 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %21 = loom.semaphore_take %20 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_3 = arith.constant 0 : index
              %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %19, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %23 = arith.addi %arg6, %arg4 : index
              loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %23], LR : [%arg5, %23]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %24 = loom.bufferize_to_tensor %21[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %25 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %26 = loom.semaphore_take %25 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %27 = loom.init_tensor %26[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
              %29 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %30 = loom.semaphore_take %29 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %31 = loom.init_tensor %30[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %32 = linalg.fill ins(%cst_0 : f16) outs(%31 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %33 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %34 = loom.semaphore_take %33 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %35 = loom.init_tensor %34[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %36 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %37 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %38 = loom.semaphore_take %37 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %39 = loom.init_tensor %38[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %40 = loom.semaphore_take %37 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %41 = loom.init_tensor %40[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %42 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %43 = loom.semaphore_take %42 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %44 = loom.init_tensor %43[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %45 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %46 = loom.semaphore_take %45 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %47 = loom.init_tensor %46[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %48 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %49 = loom.semaphore_take %48 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %50 = loom.init_tensor %49[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %51 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %52 = loom.semaphore_take %51 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %53 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %54 = loom.semaphore_take %53 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %55 = loom.init_tensor %54[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %56 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %57 = loom.semaphore_take %56 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %58 = loom.init_tensor %57[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %59 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %60 = loom.semaphore_take %59 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %61:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %36, %arg10 = %32, %arg11 = %28) -> (tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>) {
                %69 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %70 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%18, %c0_6, %69)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%70], sizes: [1, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %23], LR : [%c7, %23]) : memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x512xf16>
                %71 = loom.bufferize_to_tensor %52[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %72 = linalg.fill ins(%cst : f16) outs(%55 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                %73 = linalg.batch_matmul ins(%24, %71 : tensor<1x512x128xf16>, tensor<1x128x512xf16>) outs(%72 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                loom.semaphore_give %52 : memref<1x128x512xf16>
                %74 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%73 : tensor<1x512x512xf16>) outs(%74 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %89 = arith.maximumf %in, %out : f16
                  linalg.yield %89 : f16
                } -> tensor<1x512x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %75 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%44 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %89 = arith.mulf %in_10, %cst_2 : f16
                  %90 = arith.cmpf ogt, %in, %89 : f16
                  %91 = arith.select %90, %in, %89 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x1xf16>
                %77 = loom.broadcast ins(%76 : tensor<1x512x1xf16>) outs(%58 : tensor<1x512x512xf16>) dim(2) -> tensor<1x512x512xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %77 : tensor<1x512x512xf16>, tensor<1x512x512xf16>) outs(%55 : tensor<1x512x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %89 = arith.mulf %in, %cst_2 : f16
                  %90 = arith.subf %89, %in_10 : f16
                  %91 = math.exp %90 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x512xf16>
                loom.semaphore_give %57 : memref<1x512x512xf16>
                %79 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x512x512xf16>) outs(%79 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %89 = arith.addf %in, %out : f16
                  linalg.yield %89 : f16
                } -> tensor<1x512x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %76 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%50 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %89 = arith.subf %in, %in_10 : f16
                  %90 = math.exp %89 : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %81, %80 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%arg10 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %89 = arith.mulf %in, %in_10 : f16
                  %90 = arith.addf %89, %in_11 : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x1xf16>
                loom.semaphore_give %46 : memref<1x512x1xf16>
                %c0_8 = arith.constant 0 : index
                %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %69, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%83], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %23], LR : [%c7, %23]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %84 = loom.bufferize_to_tensor %60[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %85 = linalg.fill ins(%cst : f16) outs(%41 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                %86 = linalg.batch_matmul ins(%78, %84 : tensor<1x512x512xf16>, tensor<1x512x128xf16>) outs(%85 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                loom.semaphore_give %60 : memref<1x512x128xf16>
                loom.semaphore_give %54 : memref<1x512x512xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %arg11, %81 : tensor<1x512x128xf16>, tensor<1x512x128xf16>, tensor<1x512x1xf16>) outs(%arg11 : tensor<1x512x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %89 = arith.mulf %in_10, %in_11 : f16
                  %90 = arith.addf %in, %89 : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x128xf16>
                loom.semaphore_give %49 : memref<1x512x1xf16>
                loom.semaphore_give %40 : memref<1x512x128xf16>
                %88 = linalg.copy ins(%76 : tensor<1x512x1xf16>) outs(%arg9 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                loom.semaphore_give %43 : memref<1x512x1xf16>
                scf.yield %88, %82, %87 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>
              }
              loom.semaphore_give %34 : memref<1x512x1xf16>
              loom.semaphore_give %21 : memref<1x512x128xf16>
              %62 = loom.broadcast ins(%61#1 : tensor<1x512x1xf16>) outs(%39 : tensor<1x512x128xf16>) dim(2) -> tensor<1x512x128xf16>
              loom.semaphore_give %30 : memref<1x512x1xf16>
              %63 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %64 = loom.semaphore_take %63 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %65 = loom.init_tensor %64[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61#2, %62 : tensor<1x512x128xf16>, tensor<1x512x128xf16>) outs(%65 : tensor<1x512x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %69 = arith.divf %in, %in_6 : f16
                linalg.yield %69 : f16
              } -> tensor<1x512x128xf16>
              loom.semaphore_give %38 : memref<1x512x128xf16>
              loom.semaphore_give %26 : memref<1x512x128xf16>
              %c0_4 = arith.constant 0 : index
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %19, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%67], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %68 = loom.bufferize_to_memref %66 : tensor<1x512x128xf16> -> memref<1x512x128xf16>
              loom.copy %68, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %23], LR : [%arg5, %23]) : memref<1x512x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %64 : memref<1x512x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y2y4__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc2_dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b1__tile_m512__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
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
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %20 = arith.muli %19, %c512 : index
              %21 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %22 = loom.semaphore_take %21 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.muli %arg4, %c2 : index
              %25 = arith.addi %arg6, %24 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %26 = loom.bufferize_to_tensor %22[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %27 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %28 = loom.semaphore_take %27 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %29 = loom.init_tensor %28[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
              %31 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %32 = loom.semaphore_take %31 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %33 = loom.init_tensor %32[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %34 = linalg.fill ins(%cst_0 : f16) outs(%33 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %35 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %36 = loom.semaphore_take %35 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %37 = loom.init_tensor %36[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %38 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %39 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %40 = loom.semaphore_take %39 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %41 = loom.init_tensor %40[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %42 = loom.semaphore_take %39 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %43 = loom.init_tensor %42[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %44 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %45 = loom.semaphore_take %44 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %46 = loom.init_tensor %45[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %47 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %48 = loom.semaphore_take %47 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %49 = loom.init_tensor %48[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %50 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %51 = loom.semaphore_take %50 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %52 = loom.init_tensor %51[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %53 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %54 = loom.semaphore_take %53 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %55 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %56 = loom.semaphore_take %55 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %57 = loom.init_tensor %56[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %58 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %59 = loom.semaphore_take %58 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %60 = loom.init_tensor %59[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %61 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %62 = loom.semaphore_take %61 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %63:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %38, %arg10 = %34, %arg11 = %30) -> (tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>) {
                %71 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%18, %c0_6, %71)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [1, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                %73 = arith.addi %24, %c1 : index
                loom.copy %reinterpret_cast_7, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %24], LR : [%c7, %73]) : memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x512xf16>
                %74 = loom.bufferize_to_tensor %54[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                %76 = linalg.batch_matmul ins(%26, %74 : tensor<1x512x128xf16>, tensor<1x128x512xf16>) outs(%75 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                loom.semaphore_give %54 : memref<1x128x512xf16>
                %77 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x512x512xf16>) outs(%77 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %78 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%46 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in_10, %cst_2 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<1x512x1xf16>
                %80 = loom.broadcast ins(%79 : tensor<1x512x1xf16>) outs(%60 : tensor<1x512x512xf16>) dim(2) -> tensor<1x512x512xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %80 : tensor<1x512x512xf16>, tensor<1x512x512xf16>) outs(%57 : tensor<1x512x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_2 : f16
                  %93 = arith.subf %92, %in_10 : f16
                  %94 = math.exp %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<1x512x512xf16>
                loom.semaphore_give %59 : memref<1x512x512xf16>
                %82 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<1x512x512xf16>) outs(%82 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %79 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%52 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.subf %in, %in_10 : f16
                  %93 = math.exp %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%arg10 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %92 = arith.mulf %in, %in_10 : f16
                  %93 = arith.addf %92, %in_11 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x1xf16>
                loom.semaphore_give %48 : memref<1x512x1xf16>
                %c0_8 = arith.constant 0 : index
                %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %71, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %24], LR : [%c7, %73]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %87 = loom.bufferize_to_tensor %62[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %88 = linalg.fill ins(%cst : f16) outs(%43 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<1x512x512xf16>, tensor<1x512x128xf16>) outs(%88 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                loom.semaphore_give %62 : memref<1x512x128xf16>
                loom.semaphore_give %56 : memref<1x512x512xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<1x512x128xf16>, tensor<1x512x128xf16>, tensor<1x512x1xf16>) outs(%arg11 : tensor<1x512x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %92 = arith.mulf %in_10, %in_11 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x128xf16>
                loom.semaphore_give %51 : memref<1x512x1xf16>
                loom.semaphore_give %42 : memref<1x512x128xf16>
                %91 = linalg.copy ins(%79 : tensor<1x512x1xf16>) outs(%arg9 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                loom.semaphore_give %45 : memref<1x512x1xf16>
                scf.yield %91, %85, %90 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>
              }
              loom.semaphore_give %36 : memref<1x512x1xf16>
              loom.semaphore_give %22 : memref<1x512x128xf16>
              %64 = loom.broadcast ins(%63#1 : tensor<1x512x1xf16>) outs(%41 : tensor<1x512x128xf16>) dim(2) -> tensor<1x512x128xf16>
              loom.semaphore_give %32 : memref<1x512x1xf16>
              %65 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %66 = loom.semaphore_take %65 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %67 = loom.init_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63#2, %64 : tensor<1x512x128xf16>, tensor<1x512x128xf16>) outs(%67 : tensor<1x512x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %71 = arith.divf %in, %in_6 : f16
                linalg.yield %71 : f16
              } -> tensor<1x512x128xf16>
              loom.semaphore_give %40 : memref<1x512x128xf16>
              loom.semaphore_give %28 : memref<1x512x128xf16>
              %c0_4 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %70 = loom.bufferize_to_memref %68 : tensor<1x512x128xf16> -> memref<1x512x128xf16>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x512x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %66 : memref<1x512x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y4y2__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc4_dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b1__tile_m512__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
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
            scf.for %arg7 = %c0 to %c16 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %20 = arith.muli %19, %c512 : index
              %21 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %22 = loom.semaphore_take %21 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.muli %arg4, %c4 : index
              %25 = arith.addi %arg6, %24 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %26 = loom.bufferize_to_tensor %22[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %27 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %28 = loom.semaphore_take %27 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %29 = loom.init_tensor %28[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
              %31 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %32 = loom.semaphore_take %31 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %33 = loom.init_tensor %32[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %34 = linalg.fill ins(%cst_0 : f16) outs(%33 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %35 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %36 = loom.semaphore_take %35 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %37 = loom.init_tensor %36[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %38 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %39 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %40 = loom.semaphore_take %39 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %41 = loom.init_tensor %40[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %42 = loom.semaphore_take %39 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %43 = loom.init_tensor %42[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %44 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %45 = loom.semaphore_take %44 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %46 = loom.init_tensor %45[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %47 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %48 = loom.semaphore_take %47 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %49 = loom.init_tensor %48[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %50 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %51 = loom.semaphore_take %50 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %52 = loom.init_tensor %51[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %53 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %54 = loom.semaphore_take %53 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %55 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %56 = loom.semaphore_take %55 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %57 = loom.init_tensor %56[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %58 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %59 = loom.semaphore_take %58 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %60 = loom.init_tensor %59[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %61 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %62 = loom.semaphore_take %61 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %63:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %38, %arg10 = %34, %arg11 = %30) -> (tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>) {
                %71 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%18, %c0_6, %71)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [1, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                %73 = arith.addi %24, %c3 : index
                loom.copy %reinterpret_cast_7, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %24], LR : [%c7, %73]) : memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x512xf16>
                %74 = loom.bufferize_to_tensor %54[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                %76 = linalg.batch_matmul ins(%26, %74 : tensor<1x512x128xf16>, tensor<1x128x512xf16>) outs(%75 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                loom.semaphore_give %54 : memref<1x128x512xf16>
                %77 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x512x512xf16>) outs(%77 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %78 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%46 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in_10, %cst_2 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<1x512x1xf16>
                %80 = loom.broadcast ins(%79 : tensor<1x512x1xf16>) outs(%60 : tensor<1x512x512xf16>) dim(2) -> tensor<1x512x512xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %80 : tensor<1x512x512xf16>, tensor<1x512x512xf16>) outs(%57 : tensor<1x512x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_2 : f16
                  %93 = arith.subf %92, %in_10 : f16
                  %94 = math.exp %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<1x512x512xf16>
                loom.semaphore_give %59 : memref<1x512x512xf16>
                %82 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<1x512x512xf16>) outs(%82 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %79 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%52 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.subf %in, %in_10 : f16
                  %93 = math.exp %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%arg10 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %92 = arith.mulf %in, %in_10 : f16
                  %93 = arith.addf %92, %in_11 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x1xf16>
                loom.semaphore_give %48 : memref<1x512x1xf16>
                %c0_8 = arith.constant 0 : index
                %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %71, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %24], LR : [%c7, %73]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %87 = loom.bufferize_to_tensor %62[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %88 = linalg.fill ins(%cst : f16) outs(%43 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<1x512x512xf16>, tensor<1x512x128xf16>) outs(%88 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                loom.semaphore_give %62 : memref<1x512x128xf16>
                loom.semaphore_give %56 : memref<1x512x512xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<1x512x128xf16>, tensor<1x512x128xf16>, tensor<1x512x1xf16>) outs(%arg11 : tensor<1x512x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %92 = arith.mulf %in_10, %in_11 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x128xf16>
                loom.semaphore_give %51 : memref<1x512x1xf16>
                loom.semaphore_give %42 : memref<1x512x128xf16>
                %91 = linalg.copy ins(%79 : tensor<1x512x1xf16>) outs(%arg9 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                loom.semaphore_give %45 : memref<1x512x1xf16>
                scf.yield %91, %85, %90 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>
              }
              loom.semaphore_give %36 : memref<1x512x1xf16>
              loom.semaphore_give %22 : memref<1x512x128xf16>
              %64 = loom.broadcast ins(%63#1 : tensor<1x512x1xf16>) outs(%41 : tensor<1x512x128xf16>) dim(2) -> tensor<1x512x128xf16>
              loom.semaphore_give %32 : memref<1x512x1xf16>
              %65 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %66 = loom.semaphore_take %65 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %67 = loom.init_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63#2, %64 : tensor<1x512x128xf16>, tensor<1x512x128xf16>) outs(%67 : tensor<1x512x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %71 = arith.divf %in, %in_6 : f16
                linalg.yield %71 : f16
              } -> tensor<1x512x128xf16>
              loom.semaphore_give %40 : memref<1x512x128xf16>
              loom.semaphore_give %28 : memref<1x512x128xf16>
              %c0_4 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %70 = loom.bufferize_to_memref %68 : tensor<1x512x128xf16> -> memref<1x512x128xf16>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<1x512x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %66 : memref<1x512x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y8y1__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level1_bc8_dim_x_level0_bc8_dim_y_level1_bc8_n__tile_b1__tile_m512__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c32 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %19 = arith.muli %18, %c512 : index
              %20 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %21 = loom.semaphore_take %20 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_3 = arith.constant 0 : index
              %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %19, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %23 = arith.muli %arg4, %c8 : index
              %24 = arith.addi %arg6, %23 : index
              loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %24], LR : [%arg5, %24]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %25 = loom.bufferize_to_tensor %21[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %26 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %27 = loom.semaphore_take %26 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %28 = loom.init_tensor %27[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
              %30 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %31 = loom.semaphore_take %30 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %32 = loom.init_tensor %31[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %33 = linalg.fill ins(%cst_0 : f16) outs(%32 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %34 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %35 = loom.semaphore_take %34 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %36 = loom.init_tensor %35[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %37 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %38 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %39 = loom.semaphore_take %38 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %40 = loom.init_tensor %39[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %41 = loom.semaphore_take %38 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %42 = loom.init_tensor %41[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %43 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %44 = loom.semaphore_take %43 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %45 = loom.init_tensor %44[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %46 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %47 = loom.semaphore_take %46 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %48 = loom.init_tensor %47[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %49 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %51 = loom.init_tensor %50[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %52 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %53 = loom.semaphore_take %52 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %54 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %55 = loom.semaphore_take %54 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %56 = loom.init_tensor %55[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %57 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %58 = loom.semaphore_take %57 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %59 = loom.init_tensor %58[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %60 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %61 = loom.semaphore_take %60 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %62:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %37, %arg10 = %33, %arg11 = %29) -> (tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>) {
                %70 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%arg7, %c0_6, %70)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [1, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x512xf16>
                %72 = loom.bufferize_to_tensor %53[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %73 = linalg.fill ins(%cst : f16) outs(%56 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                %74 = linalg.batch_matmul ins(%25, %72 : tensor<1x512x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                loom.semaphore_give %53 : memref<1x128x512xf16>
                %75 = linalg.fill ins(%cst_1 : f16) outs(%45 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x512x512xf16>) outs(%75 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %90 = arith.maximumf %in, %out : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x1xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %76 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%45 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %90 = arith.mulf %in_10, %cst_2 : f16
                  %91 = arith.cmpf ogt, %in, %90 : f16
                  %92 = arith.select %91, %in, %90 : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %78 = loom.broadcast ins(%77 : tensor<1x512x1xf16>) outs(%59 : tensor<1x512x512xf16>) dim(2) -> tensor<1x512x512xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x512x512xf16>, tensor<1x512x512xf16>) outs(%56 : tensor<1x512x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %90 = arith.mulf %in, %cst_2 : f16
                  %91 = arith.subf %90, %in_10 : f16
                  %92 = math.exp %91 : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x512xf16>
                loom.semaphore_give %58 : memref<1x512x512xf16>
                %80 = linalg.fill ins(%cst : f16) outs(%48 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x512x512xf16>) outs(%80 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %90 = arith.addf %in, %out : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %77 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%51 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %90 = arith.subf %in, %in_10 : f16
                  %91 = math.exp %90 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %82, %81 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%arg10 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %90 = arith.mulf %in, %in_10 : f16
                  %91 = arith.addf %90, %in_11 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x1xf16>
                loom.semaphore_give %47 : memref<1x512x1xf16>
                %c0_8 = arith.constant 0 : index
                %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %70, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %85 = loom.bufferize_to_tensor %61[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %86 = linalg.fill ins(%cst : f16) outs(%42 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                %87 = linalg.batch_matmul ins(%79, %85 : tensor<1x512x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                loom.semaphore_give %61 : memref<1x512x128xf16>
                loom.semaphore_give %55 : memref<1x512x512xf16>
                %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %arg11, %82 : tensor<1x512x128xf16>, tensor<1x512x128xf16>, tensor<1x512x1xf16>) outs(%arg11 : tensor<1x512x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %90 = arith.mulf %in_10, %in_11 : f16
                  %91 = arith.addf %in, %90 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x128xf16>
                loom.semaphore_give %50 : memref<1x512x1xf16>
                loom.semaphore_give %41 : memref<1x512x128xf16>
                %89 = linalg.copy ins(%77 : tensor<1x512x1xf16>) outs(%arg9 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                loom.semaphore_give %44 : memref<1x512x1xf16>
                scf.yield %89, %83, %88 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>
              }
              loom.semaphore_give %35 : memref<1x512x1xf16>
              loom.semaphore_give %21 : memref<1x512x128xf16>
              %63 = loom.broadcast ins(%62#1 : tensor<1x512x1xf16>) outs(%40 : tensor<1x512x128xf16>) dim(2) -> tensor<1x512x128xf16>
              loom.semaphore_give %31 : memref<1x512x1xf16>
              %64 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %65 = loom.semaphore_take %64 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %66 = loom.init_tensor %65[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62#2, %63 : tensor<1x512x128xf16>, tensor<1x512x128xf16>) outs(%66 : tensor<1x512x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %70 = arith.divf %in, %in_6 : f16
                linalg.yield %70 : f16
              } -> tensor<1x512x128xf16>
              loom.semaphore_give %39 : memref<1x512x128xf16>
              loom.semaphore_give %27 : memref<1x512x128xf16>
              %c0_4 = arith.constant 0 : index
              %68 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %19, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%68], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %69 = loom.bufferize_to_memref %67 : tensor<1x512x128xf16> -> memref<1x512x128xf16>
              loom.copy %69, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %24], LR : [%arg5, %24]) : memref<1x512x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %65 : memref<1x512x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x1x8_y8__d0i1_d1i1_d2i0__f01__n_dim_y_level0_bc8_dim_y_level0_bc8_n__tile_b1__tile_m512__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
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
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
              %19 = arith.muli %arg6, %c512 : index
              %20 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %21 = loom.semaphore_take %20 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_3 = arith.constant 0 : index
              %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %19, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %23 = arith.addi %arg5, %arg4 : index
              loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%23, %arg6], LR : [%23, %arg6]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %24 = loom.bufferize_to_tensor %21[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %25 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %26 = loom.semaphore_take %25 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %27 = loom.init_tensor %26[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
              %29 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %30 = loom.semaphore_take %29 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %31 = loom.init_tensor %30[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %32 = linalg.fill ins(%cst_0 : f16) outs(%31 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %33 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %34 = loom.semaphore_take %33 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %35 = loom.init_tensor %34[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %36 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %37 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %38 = loom.semaphore_take %37 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %39 = loom.init_tensor %38[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %40 = loom.semaphore_take %37 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %41 = loom.init_tensor %40[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %42 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %43 = loom.semaphore_take %42 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %44 = loom.init_tensor %43[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %45 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %46 = loom.semaphore_take %45 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %47 = loom.init_tensor %46[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %48 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %49 = loom.semaphore_take %48 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %50 = loom.init_tensor %49[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %51 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %52 = loom.semaphore_take %51 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %53 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %54 = loom.semaphore_take %53 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %55 = loom.init_tensor %54[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %56 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %57 = loom.semaphore_take %56 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %58 = loom.init_tensor %57[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %59 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %60 = loom.semaphore_take %59 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %61:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %36, %arg10 = %32, %arg11 = %28) -> (tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>) {
                %69 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %70 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%18, %c0_6, %69)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%70], sizes: [1, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%23, %c0], LR : [%23, %c7]) : memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x512xf16>
                %71 = loom.bufferize_to_tensor %52[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %72 = linalg.fill ins(%cst : f16) outs(%55 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                %73 = linalg.batch_matmul ins(%24, %71 : tensor<1x512x128xf16>, tensor<1x128x512xf16>) outs(%72 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                loom.semaphore_give %52 : memref<1x128x512xf16>
                %74 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%73 : tensor<1x512x512xf16>) outs(%74 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %89 = arith.maximumf %in, %out : f16
                  linalg.yield %89 : f16
                } -> tensor<1x512x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %75 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%44 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %89 = arith.mulf %in_10, %cst_2 : f16
                  %90 = arith.cmpf ogt, %in, %89 : f16
                  %91 = arith.select %90, %in, %89 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x1xf16>
                %77 = loom.broadcast ins(%76 : tensor<1x512x1xf16>) outs(%58 : tensor<1x512x512xf16>) dim(2) -> tensor<1x512x512xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %77 : tensor<1x512x512xf16>, tensor<1x512x512xf16>) outs(%55 : tensor<1x512x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %89 = arith.mulf %in, %cst_2 : f16
                  %90 = arith.subf %89, %in_10 : f16
                  %91 = math.exp %90 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x512xf16>
                loom.semaphore_give %57 : memref<1x512x512xf16>
                %79 = linalg.fill ins(%cst : f16) outs(%47 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%78 : tensor<1x512x512xf16>) outs(%79 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %89 = arith.addf %in, %out : f16
                  linalg.yield %89 : f16
                } -> tensor<1x512x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %76 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%50 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %89 = arith.subf %in, %in_10 : f16
                  %90 = math.exp %89 : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %81, %80 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%arg10 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %89 = arith.mulf %in, %in_10 : f16
                  %90 = arith.addf %89, %in_11 : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x1xf16>
                loom.semaphore_give %46 : memref<1x512x1xf16>
                %c0_8 = arith.constant 0 : index
                %83 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %69, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%83], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%23, %c0], LR : [%23, %c7]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %84 = loom.bufferize_to_tensor %60[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %85 = linalg.fill ins(%cst : f16) outs(%41 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                %86 = linalg.batch_matmul ins(%78, %84 : tensor<1x512x512xf16>, tensor<1x512x128xf16>) outs(%85 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                loom.semaphore_give %60 : memref<1x512x128xf16>
                loom.semaphore_give %54 : memref<1x512x512xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%86, %arg11, %81 : tensor<1x512x128xf16>, tensor<1x512x128xf16>, tensor<1x512x1xf16>) outs(%arg11 : tensor<1x512x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %89 = arith.mulf %in_10, %in_11 : f16
                  %90 = arith.addf %in, %89 : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x128xf16>
                loom.semaphore_give %49 : memref<1x512x1xf16>
                loom.semaphore_give %40 : memref<1x512x128xf16>
                %88 = linalg.copy ins(%76 : tensor<1x512x1xf16>) outs(%arg9 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                loom.semaphore_give %43 : memref<1x512x1xf16>
                scf.yield %88, %82, %87 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>
              }
              loom.semaphore_give %34 : memref<1x512x1xf16>
              loom.semaphore_give %21 : memref<1x512x128xf16>
              %62 = loom.broadcast ins(%61#1 : tensor<1x512x1xf16>) outs(%39 : tensor<1x512x128xf16>) dim(2) -> tensor<1x512x128xf16>
              loom.semaphore_give %30 : memref<1x512x1xf16>
              %63 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %64 = loom.semaphore_take %63 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %65 = loom.init_tensor %64[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61#2, %62 : tensor<1x512x128xf16>, tensor<1x512x128xf16>) outs(%65 : tensor<1x512x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %69 = arith.divf %in, %in_6 : f16
                linalg.yield %69 : f16
              } -> tensor<1x512x128xf16>
              loom.semaphore_give %38 : memref<1x512x128xf16>
              loom.semaphore_give %26 : memref<1x512x128xf16>
              %c0_4 = arith.constant 0 : index
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %19, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%67], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %68 = loom.bufferize_to_memref %66 : tensor<1x512x128xf16> -> memref<1x512x128xf16>
              loom.copy %68, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%23, %arg6], LR : [%23, %arg6]) : memref<1x512x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %64 : memref<1x512x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x2x4_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc2_dim_y_level0_bc8_dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b1__tile_m512__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
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
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg4, %arg7)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg5, %arg6)
              %20 = arith.muli %19, %c512 : index
              %21 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %22 = loom.semaphore_take %21 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.muli %arg4, %c2 : index
              %25 = arith.addi %arg5, %24 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %26 = loom.bufferize_to_tensor %22[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %27 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %28 = loom.semaphore_take %27 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %29 = loom.init_tensor %28[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
              %31 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %32 = loom.semaphore_take %31 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %33 = loom.init_tensor %32[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %34 = linalg.fill ins(%cst_0 : f16) outs(%33 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %35 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %36 = loom.semaphore_take %35 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %37 = loom.init_tensor %36[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %38 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %39 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %40 = loom.semaphore_take %39 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %41 = loom.init_tensor %40[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %42 = loom.semaphore_take %39 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %43 = loom.init_tensor %42[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %44 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %45 = loom.semaphore_take %44 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %46 = loom.init_tensor %45[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %47 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %48 = loom.semaphore_take %47 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %49 = loom.init_tensor %48[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %50 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %51 = loom.semaphore_take %50 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %52 = loom.init_tensor %51[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %53 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %54 = loom.semaphore_take %53 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %55 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %56 = loom.semaphore_take %55 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %57 = loom.init_tensor %56[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %58 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %59 = loom.semaphore_take %58 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %60 = loom.init_tensor %59[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %61 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %62 = loom.semaphore_take %61 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %63:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %38, %arg10 = %34, %arg11 = %30) -> (tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>) {
                %71 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%18, %c0_6, %71)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [1, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                %73 = arith.addi %24, %c1 : index
                loom.copy %reinterpret_cast_7, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%24, %c0], LR : [%73, %c7]) : memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x512xf16>
                %74 = loom.bufferize_to_tensor %54[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                %76 = linalg.batch_matmul ins(%26, %74 : tensor<1x512x128xf16>, tensor<1x128x512xf16>) outs(%75 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                loom.semaphore_give %54 : memref<1x128x512xf16>
                %77 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x512x512xf16>) outs(%77 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %78 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%46 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in_10, %cst_2 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<1x512x1xf16>
                %80 = loom.broadcast ins(%79 : tensor<1x512x1xf16>) outs(%60 : tensor<1x512x512xf16>) dim(2) -> tensor<1x512x512xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %80 : tensor<1x512x512xf16>, tensor<1x512x512xf16>) outs(%57 : tensor<1x512x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_2 : f16
                  %93 = arith.subf %92, %in_10 : f16
                  %94 = math.exp %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<1x512x512xf16>
                loom.semaphore_give %59 : memref<1x512x512xf16>
                %82 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<1x512x512xf16>) outs(%82 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %79 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%52 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.subf %in, %in_10 : f16
                  %93 = math.exp %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%arg10 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %92 = arith.mulf %in, %in_10 : f16
                  %93 = arith.addf %92, %in_11 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x1xf16>
                loom.semaphore_give %48 : memref<1x512x1xf16>
                %c0_8 = arith.constant 0 : index
                %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %71, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%24, %c0], LR : [%73, %c7]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %87 = loom.bufferize_to_tensor %62[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %88 = linalg.fill ins(%cst : f16) outs(%43 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<1x512x512xf16>, tensor<1x512x128xf16>) outs(%88 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                loom.semaphore_give %62 : memref<1x512x128xf16>
                loom.semaphore_give %56 : memref<1x512x512xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<1x512x128xf16>, tensor<1x512x128xf16>, tensor<1x512x1xf16>) outs(%arg11 : tensor<1x512x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %92 = arith.mulf %in_10, %in_11 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x128xf16>
                loom.semaphore_give %51 : memref<1x512x1xf16>
                loom.semaphore_give %42 : memref<1x512x128xf16>
                %91 = linalg.copy ins(%79 : tensor<1x512x1xf16>) outs(%arg9 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                loom.semaphore_give %45 : memref<1x512x1xf16>
                scf.yield %91, %85, %90 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>
              }
              loom.semaphore_give %36 : memref<1x512x1xf16>
              loom.semaphore_give %22 : memref<1x512x128xf16>
              %64 = loom.broadcast ins(%63#1 : tensor<1x512x1xf16>) outs(%41 : tensor<1x512x128xf16>) dim(2) -> tensor<1x512x128xf16>
              loom.semaphore_give %32 : memref<1x512x1xf16>
              %65 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %66 = loom.semaphore_take %65 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %67 = loom.init_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63#2, %64 : tensor<1x512x128xf16>, tensor<1x512x128xf16>) outs(%67 : tensor<1x512x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %71 = arith.divf %in, %in_6 : f16
                linalg.yield %71 : f16
              } -> tensor<1x512x128xf16>
              loom.semaphore_give %40 : memref<1x512x128xf16>
              loom.semaphore_give %28 : memref<1x512x128xf16>
              %c0_4 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %70 = loom.bufferize_to_memref %68 : tensor<1x512x128xf16> -> memref<1x512x128xf16>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x512x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %66 : memref<1x512x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x4x2_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc4_dim_y_level0_bc8_dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b1__tile_m512__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
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
            scf.for %arg7 = %c0 to %c16 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg4, %arg7)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%arg5, %arg6)
              %20 = arith.muli %19, %c512 : index
              %21 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %22 = loom.semaphore_take %21 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.muli %arg4, %c4 : index
              %25 = arith.addi %arg5, %24 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %26 = loom.bufferize_to_tensor %22[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %27 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %28 = loom.semaphore_take %27 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %29 = loom.init_tensor %28[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
              %31 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %32 = loom.semaphore_take %31 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %33 = loom.init_tensor %32[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %34 = linalg.fill ins(%cst_0 : f16) outs(%33 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %35 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %36 = loom.semaphore_take %35 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %37 = loom.init_tensor %36[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %38 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %39 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %40 = loom.semaphore_take %39 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %41 = loom.init_tensor %40[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %42 = loom.semaphore_take %39 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %43 = loom.init_tensor %42[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %44 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %45 = loom.semaphore_take %44 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %46 = loom.init_tensor %45[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %47 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %48 = loom.semaphore_take %47 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %49 = loom.init_tensor %48[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %50 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %51 = loom.semaphore_take %50 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %52 = loom.init_tensor %51[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %53 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %54 = loom.semaphore_take %53 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %55 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %56 = loom.semaphore_take %55 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %57 = loom.init_tensor %56[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %58 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %59 = loom.semaphore_take %58 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %60 = loom.init_tensor %59[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %61 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %62 = loom.semaphore_take %61 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %63:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %38, %arg10 = %34, %arg11 = %30) -> (tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>) {
                %71 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%18, %c0_6, %71)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [1, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                %73 = arith.addi %24, %c3 : index
                loom.copy %reinterpret_cast_7, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%24, %c0], LR : [%73, %c7]) : memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x512xf16>
                %74 = loom.bufferize_to_tensor %54[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                %76 = linalg.batch_matmul ins(%26, %74 : tensor<1x512x128xf16>, tensor<1x128x512xf16>) outs(%75 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                loom.semaphore_give %54 : memref<1x128x512xf16>
                %77 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<1x512x512xf16>) outs(%77 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %78 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%46 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in_10, %cst_2 : f16
                  %93 = arith.cmpf ogt, %in, %92 : f16
                  %94 = arith.select %93, %in, %92 : f16
                  linalg.yield %94 : f16
                } -> tensor<1x512x1xf16>
                %80 = loom.broadcast ins(%79 : tensor<1x512x1xf16>) outs(%60 : tensor<1x512x512xf16>) dim(2) -> tensor<1x512x512xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76, %80 : tensor<1x512x512xf16>, tensor<1x512x512xf16>) outs(%57 : tensor<1x512x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_2 : f16
                  %93 = arith.subf %92, %in_10 : f16
                  %94 = math.exp %93 : f16
                  linalg.yield %94 : f16
                } -> tensor<1x512x512xf16>
                loom.semaphore_give %59 : memref<1x512x512xf16>
                %82 = linalg.fill ins(%cst : f16) outs(%49 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<1x512x512xf16>) outs(%82 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %79 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%52 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.subf %in, %in_10 : f16
                  %93 = math.exp %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84, %83 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%arg10 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %92 = arith.mulf %in, %in_10 : f16
                  %93 = arith.addf %92, %in_11 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x1xf16>
                loom.semaphore_give %48 : memref<1x512x1xf16>
                %c0_8 = arith.constant 0 : index
                %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %71, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%24, %c0], LR : [%73, %c7]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %87 = loom.bufferize_to_tensor %62[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %88 = linalg.fill ins(%cst : f16) outs(%43 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<1x512x512xf16>, tensor<1x512x128xf16>) outs(%88 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                loom.semaphore_give %62 : memref<1x512x128xf16>
                loom.semaphore_give %56 : memref<1x512x512xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %arg11, %84 : tensor<1x512x128xf16>, tensor<1x512x128xf16>, tensor<1x512x1xf16>) outs(%arg11 : tensor<1x512x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %92 = arith.mulf %in_10, %in_11 : f16
                  %93 = arith.addf %in, %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<1x512x128xf16>
                loom.semaphore_give %51 : memref<1x512x1xf16>
                loom.semaphore_give %42 : memref<1x512x128xf16>
                %91 = linalg.copy ins(%79 : tensor<1x512x1xf16>) outs(%arg9 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                loom.semaphore_give %45 : memref<1x512x1xf16>
                scf.yield %91, %85, %90 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>
              }
              loom.semaphore_give %36 : memref<1x512x1xf16>
              loom.semaphore_give %22 : memref<1x512x128xf16>
              %64 = loom.broadcast ins(%63#1 : tensor<1x512x1xf16>) outs(%41 : tensor<1x512x128xf16>) dim(2) -> tensor<1x512x128xf16>
              loom.semaphore_give %32 : memref<1x512x1xf16>
              %65 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %66 = loom.semaphore_take %65 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %67 = loom.init_tensor %66[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63#2, %64 : tensor<1x512x128xf16>, tensor<1x512x128xf16>) outs(%67 : tensor<1x512x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %71 = arith.divf %in, %in_6 : f16
                linalg.yield %71 : f16
              } -> tensor<1x512x128xf16>
              loom.semaphore_give %40 : memref<1x512x128xf16>
              loom.semaphore_give %28 : memref<1x512x128xf16>
              %c0_4 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%18, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %70 = loom.bufferize_to_memref %68 : tensor<1x512x128xf16> -> memref<1x512x128xf16>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<1x512x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %66 : memref<1x512x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8x1_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level1_bc8_dim_y_level0_bc8_dim_x_level1_bc8_dim_y_level0_bc8_n__tile_b1__tile_m512__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c32 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
              %19 = arith.muli %18, %c512 : index
              %20 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %21 = loom.semaphore_take %20 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %c0_3 = arith.constant 0 : index
              %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %19, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %23 = arith.muli %arg4, %c8 : index
              %24 = arith.addi %arg5, %23 : index
              loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%24, %arg6], LR : [%24, %arg6]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
              %25 = loom.bufferize_to_tensor %21[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %26 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %27 = loom.semaphore_take %26 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %28 = loom.init_tensor %27[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
              %30 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %31 = loom.semaphore_take %30 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %32 = loom.init_tensor %31[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %33 = linalg.fill ins(%cst_0 : f16) outs(%32 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %34 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %35 = loom.semaphore_take %34 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %36 = loom.init_tensor %35[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %37 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
              %38 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %39 = loom.semaphore_take %38 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %40 = loom.init_tensor %39[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %41 = loom.semaphore_take %38 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %42 = loom.init_tensor %41[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %43 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %44 = loom.semaphore_take %43 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %45 = loom.init_tensor %44[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %46 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %47 = loom.semaphore_take %46 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %48 = loom.init_tensor %47[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %49 = loom.alloc [1, 512, 1] on @L1 : memref<1x512x1xf16>
              %50 = loom.semaphore_take %49 : memref<1x512x1xf16> -> memref<1x512x1xf16>
              %51 = loom.init_tensor %50[1, 512, 1] : memref<1x512x1xf16> -> tensor<1x512x1xf16>
              %52 = loom.alloc [1, 128, 512] on @L1 : memref<1x128x512xf16>
              %53 = loom.semaphore_take %52 : memref<1x128x512xf16> -> memref<1x128x512xf16>
              %54 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %55 = loom.semaphore_take %54 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %56 = loom.init_tensor %55[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %57 = loom.alloc [1, 512, 512] on @L1 : memref<1x512x512xf16>
              %58 = loom.semaphore_take %57 : memref<1x512x512xf16> -> memref<1x512x512xf16>
              %59 = loom.init_tensor %58[1, 512, 512] : memref<1x512x512xf16> -> tensor<1x512x512xf16>
              %60 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %61 = loom.semaphore_take %60 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %62:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %37, %arg10 = %33, %arg11 = %29) -> (tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>) {
                %70 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%arg7, %c0_6, %70)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [1, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x512xf16>
                %72 = loom.bufferize_to_tensor %53[1, 128, 512] : memref<1x128x512xf16> -> tensor<1x128x512xf16>
                %73 = linalg.fill ins(%cst : f16) outs(%56 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                %74 = linalg.batch_matmul ins(%25, %72 : tensor<1x512x128xf16>, tensor<1x128x512xf16>) outs(%73 : tensor<1x512x512xf16>) -> tensor<1x512x512xf16>
                loom.semaphore_give %53 : memref<1x128x512xf16>
                %75 = linalg.fill ins(%cst_1 : f16) outs(%45 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x512x512xf16>) outs(%75 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %90 = arith.maximumf %in, %out : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x1xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %76 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%45 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %90 = arith.mulf %in_10, %cst_2 : f16
                  %91 = arith.cmpf ogt, %in, %90 : f16
                  %92 = arith.select %91, %in, %90 : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x1xf16>
                %78 = loom.broadcast ins(%77 : tensor<1x512x1xf16>) outs(%59 : tensor<1x512x512xf16>) dim(2) -> tensor<1x512x512xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74, %78 : tensor<1x512x512xf16>, tensor<1x512x512xf16>) outs(%56 : tensor<1x512x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %90 = arith.mulf %in, %cst_2 : f16
                  %91 = arith.subf %90, %in_10 : f16
                  %92 = math.exp %91 : f16
                  linalg.yield %92 : f16
                } -> tensor<1x512x512xf16>
                loom.semaphore_give %58 : memref<1x512x512xf16>
                %80 = linalg.fill ins(%cst : f16) outs(%48 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%79 : tensor<1x512x512xf16>) outs(%80 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %90 = arith.addf %in, %out : f16
                  linalg.yield %90 : f16
                } -> tensor<1x512x1xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %77 : tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%51 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %90 = arith.subf %in, %in_10 : f16
                  %91 = math.exp %90 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %82, %81 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x1xf16>) outs(%arg10 : tensor<1x512x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %90 = arith.mulf %in, %in_10 : f16
                  %91 = arith.addf %90, %in_11 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x1xf16>
                loom.semaphore_give %47 : memref<1x512x1xf16>
                %c0_8 = arith.constant 0 : index
                %84 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %70, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%84], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x512x128xf16>
                %85 = loom.bufferize_to_tensor %61[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
                %86 = linalg.fill ins(%cst : f16) outs(%42 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                %87 = linalg.batch_matmul ins(%79, %85 : tensor<1x512x512xf16>, tensor<1x512x128xf16>) outs(%86 : tensor<1x512x128xf16>) -> tensor<1x512x128xf16>
                loom.semaphore_give %61 : memref<1x512x128xf16>
                loom.semaphore_give %55 : memref<1x512x512xf16>
                %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %arg11, %82 : tensor<1x512x128xf16>, tensor<1x512x128xf16>, tensor<1x512x1xf16>) outs(%arg11 : tensor<1x512x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %in_11: f16, %out: f16):
                  %90 = arith.mulf %in_10, %in_11 : f16
                  %91 = arith.addf %in, %90 : f16
                  linalg.yield %91 : f16
                } -> tensor<1x512x128xf16>
                loom.semaphore_give %50 : memref<1x512x1xf16>
                loom.semaphore_give %41 : memref<1x512x128xf16>
                %89 = linalg.copy ins(%77 : tensor<1x512x1xf16>) outs(%arg9 : tensor<1x512x1xf16>) -> tensor<1x512x1xf16>
                loom.semaphore_give %44 : memref<1x512x1xf16>
                scf.yield %89, %83, %88 : tensor<1x512x1xf16>, tensor<1x512x1xf16>, tensor<1x512x128xf16>
              }
              loom.semaphore_give %35 : memref<1x512x1xf16>
              loom.semaphore_give %21 : memref<1x512x128xf16>
              %63 = loom.broadcast ins(%62#1 : tensor<1x512x1xf16>) outs(%40 : tensor<1x512x128xf16>) dim(2) -> tensor<1x512x128xf16>
              loom.semaphore_give %31 : memref<1x512x1xf16>
              %64 = loom.alloc [1, 512, 128] on @L1 : memref<1x512x128xf16>
              %65 = loom.semaphore_take %64 : memref<1x512x128xf16> -> memref<1x512x128xf16>
              %66 = loom.init_tensor %65[1, 512, 128] : memref<1x512x128xf16> -> tensor<1x512x128xf16>
              %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62#2, %63 : tensor<1x512x128xf16>, tensor<1x512x128xf16>) outs(%66 : tensor<1x512x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %70 = arith.divf %in, %in_6 : f16
                linalg.yield %70 : f16
              } -> tensor<1x512x128xf16>
              loom.semaphore_give %39 : memref<1x512x128xf16>
              loom.semaphore_give %27 : memref<1x512x128xf16>
              %c0_4 = arith.constant 0 : index
              %68 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%arg7, %19, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%68], sizes: [1, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              %69 = loom.bufferize_to_memref %67 : tensor<1x512x128xf16> -> memref<1x512x128xf16>
              loom.copy %69, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%24, %arg6], LR : [%24, %arg6]) : memref<1x512x128xf16> to memref<1x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %65 : memref<1x512x128xf16>
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
