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
    func.func @attention__x8_y1y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_x_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (1) {
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
              %19 = arith.muli %arg4, %c64 : index
              %20 = arith.muli %18, %c64 : index
              %21 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %22 = loom.semaphore_take %21 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.addi %arg6, %arg4 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %24], LR : [%arg5, %24]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
              %25 = loom.bufferize_to_tensor %22[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %26 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %27 = loom.semaphore_take %26 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %28 = loom.init_tensor %27[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              %30 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %31 = loom.semaphore_take %30 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %32 = loom.init_tensor %31[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %33 = linalg.fill ins(%cst_0 : f16) outs(%32 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %34 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %35 = loom.semaphore_take %34 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %36 = loom.init_tensor %35[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %37 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %38 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %39 = loom.semaphore_take %38 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %40 = loom.init_tensor %39[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %41 = loom.semaphore_take %38 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %42 = loom.init_tensor %41[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %43 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %44 = loom.semaphore_take %43 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %45 = loom.init_tensor %44[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %46 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %47 = loom.semaphore_take %46 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %48 = loom.init_tensor %47[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %49 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %50 = loom.semaphore_take %49 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %51 = loom.init_tensor %50[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %52 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
              %53 = loom.semaphore_take %52 : memref<64x128x512xf16> -> memref<64x128x512xf16>
              %54 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %55 = loom.semaphore_take %54 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %56 = loom.init_tensor %55[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %57 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %58 = loom.semaphore_take %57 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %59 = loom.init_tensor %58[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %60 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
              %61 = loom.semaphore_take %60 : memref<64x512x128xf16> -> memref<64x512x128xf16>
              %62:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %37, %arg10 = %33, %arg11 = %29) -> (tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>) {
                %70 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%19, %c0_6, %70)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %24], LR : [%c7, %24]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
                %72 = loom.bufferize_to_tensor %53[64, 128, 512] : memref<64x128x512xf16> -> tensor<64x128x512xf16>
                %73 = linalg.fill ins(%cst : f16) outs(%56 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                %74 = linalg.batch_matmul ins(%25, %72 : tensor<64x64x128xf16>, tensor<64x128x512xf16>) outs(%73 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                loom.semaphore_give %53 : memref<64x128x512xf16>
                %75 = linalg.fill ins(%cst_1 : f16) outs(%48 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<64x64x512xf16>) outs(%75 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x1xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<64x64x1xf16>) outs(%48 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %77 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%48 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.cmpf ogt, %in, %in_10 : f16
                  %93 = arith.select %92, %in, %in_10 : f16
                  linalg.yield %93 : f16
                } -> tensor<64x64x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74 : tensor<64x64x512xf16>) outs(%56 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x512xf16>
                %80 = loom.broadcast ins(%78 : tensor<64x64x1xf16>) outs(%59 : tensor<64x64x512xf16>) dim(2) -> tensor<64x64x512xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %80 : tensor<64x64x512xf16>, tensor<64x64x512xf16>) outs(%56 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.subf %in, %in_10 : f16
                  %93 = math.exp %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<64x64x512xf16>
                loom.semaphore_give %58 : memref<64x64x512xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %78 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%51 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.subf %in, %in_10 : f16
                  %93 = math.exp %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<64x64x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %82 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%arg10 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in, %in_10 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x1xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<64x64x512xf16>) outs(%83 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %82 : tensor<64x64x128xf16>, tensor<64x64x1xf16>) outs(%42 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in, %in_10 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %50 : memref<64x64x1xf16>
                %c0_8 = arith.constant 0 : index
                %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %70, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %24], LR : [%c7, %24]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
                %87 = loom.bufferize_to_tensor %61[64, 512, 128] : memref<64x512x128xf16> -> tensor<64x512x128xf16>
                %88 = linalg.fill ins(%cst : f16) outs(%45 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<64x64x512xf16>, tensor<64x512x128xf16>) outs(%88 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                loom.semaphore_give %61 : memref<64x512x128xf16>
                loom.semaphore_give %55 : memref<64x64x512xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %85 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%arg11 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.addf %in, %in_10 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %44 : memref<64x64x128xf16>
                loom.semaphore_give %41 : memref<64x64x128xf16>
                %91 = linalg.copy ins(%78 : tensor<64x64x1xf16>) outs(%arg9 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                loom.semaphore_give %47 : memref<64x64x1xf16>
                scf.yield %91, %84, %90 : tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>
              }
              loom.semaphore_give %35 : memref<64x64x1xf16>
              loom.semaphore_give %22 : memref<64x64x128xf16>
              %63 = loom.broadcast ins(%62#1 : tensor<64x64x1xf16>) outs(%40 : tensor<64x64x128xf16>) dim(2) -> tensor<64x64x128xf16>
              loom.semaphore_give %31 : memref<64x64x1xf16>
              %64 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %65 = loom.semaphore_take %64 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %66 = loom.init_tensor %65[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62#2, %63 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%66 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %70 = arith.divf %in, %in_6 : f16
                linalg.yield %70 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %39 : memref<64x64x128xf16>
              loom.semaphore_give %27 : memref<64x64x128xf16>
              %c0_4 = arith.constant 0 : index
              %68 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%68], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %69 = loom.bufferize_to_memref %67 : tensor<64x64x128xf16> -> memref<64x64x128xf16>
              loom.copy %69, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %24], LR : [%arg5, %24]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %65 : memref<64x64x128xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y2y4__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc2_dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (2) {
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg5, %arg6, %arg7)
              %19 = arith.muli %arg4, %c64 : index
              %20 = arith.muli %18, %c64 : index
              %21 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %22 = loom.semaphore_take %21 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.muli %arg4, %c2 : index
              %25 = arith.addi %arg6, %24 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
              %26 = loom.bufferize_to_tensor %22[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %27 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %28 = loom.semaphore_take %27 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %29 = loom.init_tensor %28[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              %31 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %32 = loom.semaphore_take %31 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %33 = loom.init_tensor %32[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %34 = linalg.fill ins(%cst_0 : f16) outs(%33 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %35 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %36 = loom.semaphore_take %35 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %37 = loom.init_tensor %36[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %38 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %39 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %40 = loom.semaphore_take %39 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %41 = loom.init_tensor %40[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %42 = loom.semaphore_take %39 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %43 = loom.init_tensor %42[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %44 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %45 = loom.semaphore_take %44 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %46 = loom.init_tensor %45[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %47 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %48 = loom.semaphore_take %47 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %49 = loom.init_tensor %48[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %50 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %51 = loom.semaphore_take %50 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %52 = loom.init_tensor %51[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %53 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
              %54 = loom.semaphore_take %53 : memref<64x128x512xf16> -> memref<64x128x512xf16>
              %55 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %56 = loom.semaphore_take %55 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %57 = loom.init_tensor %56[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %58 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %59 = loom.semaphore_take %58 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %60 = loom.init_tensor %59[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %61 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
              %62 = loom.semaphore_take %61 : memref<64x512x128xf16> -> memref<64x512x128xf16>
              %63:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %38, %arg10 = %34, %arg11 = %30) -> (tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>) {
                %71 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%19, %c0_6, %71)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                %73 = arith.addi %24, %c1 : index
                loom.copy %reinterpret_cast_7, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %24], LR : [%c7, %73]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
                %74 = loom.bufferize_to_tensor %54[64, 128, 512] : memref<64x128x512xf16> -> tensor<64x128x512xf16>
                %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                %76 = linalg.batch_matmul ins(%26, %74 : tensor<64x64x128xf16>, tensor<64x128x512xf16>) outs(%75 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                loom.semaphore_give %54 : memref<64x128x512xf16>
                %77 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<64x64x512xf16>) outs(%77 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.maximumf %in, %out : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78 : tensor<64x64x1xf16>) outs(%49 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %79 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%49 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.cmpf ogt, %in, %in_10 : f16
                  %95 = arith.select %94, %in, %in_10 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<64x64x512xf16>) outs(%57 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x512xf16>
                %82 = loom.broadcast ins(%80 : tensor<64x64x1xf16>) outs(%60 : tensor<64x64x512xf16>) dim(2) -> tensor<64x64x512xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %82 : tensor<64x64x512xf16>, tensor<64x64x512xf16>) outs(%57 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.subf %in, %in_10 : f16
                  %95 = math.exp %94 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x512xf16>
                loom.semaphore_give %59 : memref<64x64x512xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %80 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%52 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.subf %in, %in_10 : f16
                  %95 = math.exp %94 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%arg10 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.mulf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%83 : tensor<64x64x512xf16>) outs(%85 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.addf %in, %out : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %84 : tensor<64x64x128xf16>, tensor<64x64x1xf16>) outs(%43 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.mulf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %51 : memref<64x64x1xf16>
                %c0_8 = arith.constant 0 : index
                %88 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %71, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%88], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %24], LR : [%c7, %73]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
                %89 = loom.bufferize_to_tensor %62[64, 512, 128] : memref<64x512x128xf16> -> tensor<64x512x128xf16>
                %90 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                %91 = linalg.batch_matmul ins(%83, %89 : tensor<64x64x512xf16>, tensor<64x512x128xf16>) outs(%90 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                loom.semaphore_give %62 : memref<64x512x128xf16>
                loom.semaphore_give %56 : memref<64x64x512xf16>
                %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %87 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%arg11 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.addf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %45 : memref<64x64x128xf16>
                loom.semaphore_give %42 : memref<64x64x128xf16>
                %93 = linalg.copy ins(%80 : tensor<64x64x1xf16>) outs(%arg9 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                loom.semaphore_give %48 : memref<64x64x1xf16>
                scf.yield %93, %86, %92 : tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>
              }
              loom.semaphore_give %36 : memref<64x64x1xf16>
              loom.semaphore_give %22 : memref<64x64x128xf16>
              %64 = loom.broadcast ins(%63#1 : tensor<64x64x1xf16>) outs(%41 : tensor<64x64x128xf16>) dim(2) -> tensor<64x64x128xf16>
              loom.semaphore_give %32 : memref<64x64x1xf16>
              %65 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %66 = loom.semaphore_take %65 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %67 = loom.init_tensor %66[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63#2, %64 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%67 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %71 = arith.divf %in, %in_6 : f16
                linalg.yield %71 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %40 : memref<64x64x128xf16>
              loom.semaphore_give %28 : memref<64x64x128xf16>
              %c0_4 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %70 = loom.bufferize_to_memref %68 : tensor<64x64x128xf16> -> memref<64x64x128xf16>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %66 : memref<64x64x128xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y4y2__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc4_dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (4) {
            scf.for %arg7 = %c0 to %c2 step %c1 {
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg5, %arg6, %arg7)
              %19 = arith.muli %arg4, %c64 : index
              %20 = arith.muli %18, %c64 : index
              %21 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %22 = loom.semaphore_take %21 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.muli %arg4, %c4 : index
              %25 = arith.addi %arg6, %24 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
              %26 = loom.bufferize_to_tensor %22[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %27 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %28 = loom.semaphore_take %27 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %29 = loom.init_tensor %28[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              %31 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %32 = loom.semaphore_take %31 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %33 = loom.init_tensor %32[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %34 = linalg.fill ins(%cst_0 : f16) outs(%33 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %35 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %36 = loom.semaphore_take %35 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %37 = loom.init_tensor %36[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %38 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %39 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %40 = loom.semaphore_take %39 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %41 = loom.init_tensor %40[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %42 = loom.semaphore_take %39 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %43 = loom.init_tensor %42[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %44 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %45 = loom.semaphore_take %44 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %46 = loom.init_tensor %45[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %47 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %48 = loom.semaphore_take %47 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %49 = loom.init_tensor %48[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %50 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %51 = loom.semaphore_take %50 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %52 = loom.init_tensor %51[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %53 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
              %54 = loom.semaphore_take %53 : memref<64x128x512xf16> -> memref<64x128x512xf16>
              %55 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %56 = loom.semaphore_take %55 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %57 = loom.init_tensor %56[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %58 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %59 = loom.semaphore_take %58 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %60 = loom.init_tensor %59[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %61 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
              %62 = loom.semaphore_take %61 : memref<64x512x128xf16> -> memref<64x512x128xf16>
              %63:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %38, %arg10 = %34, %arg11 = %30) -> (tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>) {
                %71 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%19, %c0_6, %71)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                %73 = arith.addi %24, %c3 : index
                loom.copy %reinterpret_cast_7, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %24], LR : [%c7, %73]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
                %74 = loom.bufferize_to_tensor %54[64, 128, 512] : memref<64x128x512xf16> -> tensor<64x128x512xf16>
                %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                %76 = linalg.batch_matmul ins(%26, %74 : tensor<64x64x128xf16>, tensor<64x128x512xf16>) outs(%75 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                loom.semaphore_give %54 : memref<64x128x512xf16>
                %77 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<64x64x512xf16>) outs(%77 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.maximumf %in, %out : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78 : tensor<64x64x1xf16>) outs(%49 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %79 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%49 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.cmpf ogt, %in, %in_10 : f16
                  %95 = arith.select %94, %in, %in_10 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<64x64x512xf16>) outs(%57 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x512xf16>
                %82 = loom.broadcast ins(%80 : tensor<64x64x1xf16>) outs(%60 : tensor<64x64x512xf16>) dim(2) -> tensor<64x64x512xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %82 : tensor<64x64x512xf16>, tensor<64x64x512xf16>) outs(%57 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.subf %in, %in_10 : f16
                  %95 = math.exp %94 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x512xf16>
                loom.semaphore_give %59 : memref<64x64x512xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %80 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%52 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.subf %in, %in_10 : f16
                  %95 = math.exp %94 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%arg10 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.mulf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%83 : tensor<64x64x512xf16>) outs(%85 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.addf %in, %out : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %84 : tensor<64x64x128xf16>, tensor<64x64x1xf16>) outs(%43 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.mulf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %51 : memref<64x64x1xf16>
                %c0_8 = arith.constant 0 : index
                %88 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %71, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%88], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %24], LR : [%c7, %73]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
                %89 = loom.bufferize_to_tensor %62[64, 512, 128] : memref<64x512x128xf16> -> tensor<64x512x128xf16>
                %90 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                %91 = linalg.batch_matmul ins(%83, %89 : tensor<64x64x512xf16>, tensor<64x512x128xf16>) outs(%90 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                loom.semaphore_give %62 : memref<64x512x128xf16>
                loom.semaphore_give %56 : memref<64x64x512xf16>
                %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %87 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%arg11 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.addf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %45 : memref<64x64x128xf16>
                loom.semaphore_give %42 : memref<64x64x128xf16>
                %93 = linalg.copy ins(%80 : tensor<64x64x1xf16>) outs(%arg9 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                loom.semaphore_give %48 : memref<64x64x1xf16>
                scf.yield %93, %86, %92 : tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>
              }
              loom.semaphore_give %36 : memref<64x64x1xf16>
              loom.semaphore_give %22 : memref<64x64x128xf16>
              %64 = loom.broadcast ins(%63#1 : tensor<64x64x1xf16>) outs(%41 : tensor<64x64x128xf16>) dim(2) -> tensor<64x64x128xf16>
              loom.semaphore_give %32 : memref<64x64x1xf16>
              %65 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %66 = loom.semaphore_take %65 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %67 = loom.init_tensor %66[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63#2, %64 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%67 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %71 = arith.divf %in, %in_6 : f16
                linalg.yield %71 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %40 : memref<64x64x128xf16>
              loom.semaphore_give %28 : memref<64x64x128xf16>
              %c0_4 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %70 = loom.bufferize_to_memref %68 : tensor<64x64x128xf16> -> memref<64x64x128xf16>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %25], LR : [%arg5, %25]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %66 : memref<64x64x128xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y8y1__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level1_bc8_dim_x_level0_bc8_dim_y_level1_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
            %19 = arith.muli %18, %c64 : index
            %20 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %21 = loom.semaphore_take %20 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%c0_3, %19, %c0_4)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            %23 = arith.muli %arg4, %c8 : index
            %24 = arith.addi %arg6, %23 : index
            loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg5, %24], LR : [%arg5, %24]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
            %25 = loom.bufferize_to_tensor %21[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %26 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %28 = loom.init_tensor %27[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
            %30 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
            %31 = loom.semaphore_take %30 : memref<64x64x1xf16> -> memref<64x64x1xf16>
            %32 = loom.init_tensor %31[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
            %33 = linalg.fill ins(%cst_0 : f16) outs(%32 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
            %34 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
            %35 = loom.semaphore_take %34 : memref<64x64x1xf16> -> memref<64x64x1xf16>
            %36 = loom.init_tensor %35[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
            %37 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
            %38 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %39 = loom.semaphore_take %38 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %40 = loom.init_tensor %39[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %41 = loom.semaphore_take %38 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %42 = loom.init_tensor %41[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %43 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %44 = loom.semaphore_take %43 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %45 = loom.init_tensor %44[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %46 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
            %47 = loom.semaphore_take %46 : memref<64x64x1xf16> -> memref<64x64x1xf16>
            %48 = loom.init_tensor %47[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
            %49 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
            %50 = loom.semaphore_take %49 : memref<64x64x1xf16> -> memref<64x64x1xf16>
            %51 = loom.init_tensor %50[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
            %52 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
            %53 = loom.semaphore_take %52 : memref<64x128x512xf16> -> memref<64x128x512xf16>
            %54 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
            %55 = loom.semaphore_take %54 : memref<64x64x512xf16> -> memref<64x64x512xf16>
            %56 = loom.init_tensor %55[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
            %57 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
            %58 = loom.semaphore_take %57 : memref<64x64x512xf16> -> memref<64x64x512xf16>
            %59 = loom.init_tensor %58[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
            %60 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
            %61 = loom.semaphore_take %60 : memref<64x512x128xf16> -> memref<64x512x128xf16>
            %62:3 = scf.for %arg7 = %c0 to %c8 step %c1 iter_args(%arg8 = %37, %arg9 = %33, %arg10 = %29) -> (tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>) {
              %70 = arith.muli %arg7, %c512 : index
              %c0_8 = arith.constant 0 : index
              %c0_9 = arith.constant 0 : index
              %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%c0_8, %c0_9, %70)
              %reinterpret_cast_10 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_10, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
              %72 = loom.bufferize_to_tensor %53[64, 128, 512] : memref<64x128x512xf16> -> tensor<64x128x512xf16>
              %73 = linalg.fill ins(%cst : f16) outs(%56 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
              %74 = linalg.batch_matmul ins(%25, %72 : tensor<64x64x128xf16>, tensor<64x128x512xf16>) outs(%73 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
              loom.semaphore_give %53 : memref<64x128x512xf16>
              %75 = linalg.fill ins(%cst_1 : f16) outs(%48 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<64x64x512xf16>) outs(%75 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %92 = arith.maximumf %in, %out : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<64x64x1xf16>) outs(%48 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %92 = arith.mulf %in, %cst_2 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %77 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%48 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.cmpf ogt, %in, %in_14 : f16
                %93 = arith.select %92, %in, %in_14 : f16
                linalg.yield %93 : f16
              } -> tensor<64x64x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74 : tensor<64x64x512xf16>) outs(%56 : tensor<64x64x512xf16>) {
              ^bb0(%in: f16, %out: f16):
                %92 = arith.mulf %in, %cst_2 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x512xf16>
              %80 = loom.broadcast ins(%78 : tensor<64x64x1xf16>) outs(%59 : tensor<64x64x512xf16>) dim(2) -> tensor<64x64x512xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %80 : tensor<64x64x512xf16>, tensor<64x64x512xf16>) outs(%56 : tensor<64x64x512xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.subf %in, %in_14 : f16
                %93 = math.exp %92 : f16
                linalg.yield %93 : f16
              } -> tensor<64x64x512xf16>
              loom.semaphore_give %58 : memref<64x64x512xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %78 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%51 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.subf %in, %in_14 : f16
                %93 = math.exp %92 : f16
                linalg.yield %93 : f16
              } -> tensor<64x64x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %82 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%arg9 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.mulf %in, %in_14 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<64x64x512xf16>) outs(%83 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %92 = arith.addf %in, %out : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x1xf16>
              %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %82 : tensor<64x64x128xf16>, tensor<64x64x1xf16>) outs(%42 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.mulf %in, %in_14 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %50 : memref<64x64x1xf16>
              %c0_11 = arith.constant 0 : index
              %c0_12 = arith.constant 0 : index
              %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%c0_11, %70, %c0_12)
              %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_13, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
              %87 = loom.bufferize_to_tensor %61[64, 512, 128] : memref<64x512x128xf16> -> tensor<64x512x128xf16>
              %88 = linalg.fill ins(%cst : f16) outs(%45 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              %89 = linalg.batch_matmul ins(%81, %87 : tensor<64x64x512xf16>, tensor<64x512x128xf16>) outs(%88 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              loom.semaphore_give %61 : memref<64x512x128xf16>
              loom.semaphore_give %55 : memref<64x64x512xf16>
              %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %85 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%arg10 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.addf %in, %in_14 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %44 : memref<64x64x128xf16>
              loom.semaphore_give %41 : memref<64x64x128xf16>
              %91 = linalg.copy ins(%78 : tensor<64x64x1xf16>) outs(%arg8 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              loom.semaphore_give %47 : memref<64x64x1xf16>
              scf.yield %91, %84, %90 : tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>
            }
            loom.semaphore_give %35 : memref<64x64x1xf16>
            loom.semaphore_give %21 : memref<64x64x128xf16>
            %63 = loom.broadcast ins(%62#1 : tensor<64x64x1xf16>) outs(%40 : tensor<64x64x128xf16>) dim(2) -> tensor<64x64x128xf16>
            loom.semaphore_give %31 : memref<64x64x1xf16>
            %64 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %65 = loom.semaphore_take %64 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %66 = loom.init_tensor %65[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62#2, %63 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%66 : tensor<64x64x128xf16>) {
            ^bb0(%in: f16, %in_8: f16, %out: f16):
              %70 = arith.divf %in, %in_8 : f16
              linalg.yield %70 : f16
            } -> tensor<64x64x128xf16>
            loom.semaphore_give %39 : memref<64x64x128xf16>
            loom.semaphore_give %27 : memref<64x64x128xf16>
            %c0_5 = arith.constant 0 : index
            %c0_6 = arith.constant 0 : index
            %68 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%c0_5, %19, %c0_6)
            %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%68], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            %69 = loom.bufferize_to_memref %67 : tensor<64x64x128xf16> -> memref<64x64x128xf16>
            loom.copy %69, %reinterpret_cast_7 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg5, %24], LR : [%arg5, %24]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %65 : memref<64x64x128xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x1x8_y8__d0i1_d1i1_d2i0__f01__n_dim_y_level0_bc8_dim_y_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (1) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c8 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg6, %arg7)
              %19 = arith.muli %arg4, %c64 : index
              %20 = arith.muli %18, %c64 : index
              %21 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %22 = loom.semaphore_take %21 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.addi %arg5, %arg4 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%24, %arg6], LR : [%24, %arg6]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
              %25 = loom.bufferize_to_tensor %22[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %26 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %27 = loom.semaphore_take %26 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %28 = loom.init_tensor %27[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              %30 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %31 = loom.semaphore_take %30 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %32 = loom.init_tensor %31[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %33 = linalg.fill ins(%cst_0 : f16) outs(%32 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %34 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %35 = loom.semaphore_take %34 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %36 = loom.init_tensor %35[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %37 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %38 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %39 = loom.semaphore_take %38 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %40 = loom.init_tensor %39[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %41 = loom.semaphore_take %38 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %42 = loom.init_tensor %41[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %43 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %44 = loom.semaphore_take %43 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %45 = loom.init_tensor %44[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %46 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %47 = loom.semaphore_take %46 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %48 = loom.init_tensor %47[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %49 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %50 = loom.semaphore_take %49 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %51 = loom.init_tensor %50[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %52 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
              %53 = loom.semaphore_take %52 : memref<64x128x512xf16> -> memref<64x128x512xf16>
              %54 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %55 = loom.semaphore_take %54 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %56 = loom.init_tensor %55[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %57 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %58 = loom.semaphore_take %57 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %59 = loom.init_tensor %58[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %60 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
              %61 = loom.semaphore_take %60 : memref<64x512x128xf16> -> memref<64x512x128xf16>
              %62:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %37, %arg10 = %33, %arg11 = %29) -> (tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>) {
                %70 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%19, %c0_6, %70)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%24, %c0], LR : [%24, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
                %72 = loom.bufferize_to_tensor %53[64, 128, 512] : memref<64x128x512xf16> -> tensor<64x128x512xf16>
                %73 = linalg.fill ins(%cst : f16) outs(%56 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                %74 = linalg.batch_matmul ins(%25, %72 : tensor<64x64x128xf16>, tensor<64x128x512xf16>) outs(%73 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                loom.semaphore_give %53 : memref<64x128x512xf16>
                %75 = linalg.fill ins(%cst_1 : f16) outs(%48 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<64x64x512xf16>) outs(%75 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.maximumf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x1xf16>
                %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<64x64x1xf16>) outs(%48 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %77 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%48 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.cmpf ogt, %in, %in_10 : f16
                  %93 = arith.select %92, %in, %in_10 : f16
                  linalg.yield %93 : f16
                } -> tensor<64x64x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74 : tensor<64x64x512xf16>) outs(%56 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x512xf16>
                %80 = loom.broadcast ins(%78 : tensor<64x64x1xf16>) outs(%59 : tensor<64x64x512xf16>) dim(2) -> tensor<64x64x512xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %80 : tensor<64x64x512xf16>, tensor<64x64x512xf16>) outs(%56 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.subf %in, %in_10 : f16
                  %93 = math.exp %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<64x64x512xf16>
                loom.semaphore_give %58 : memref<64x64x512xf16>
                %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %78 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%51 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.subf %in, %in_10 : f16
                  %93 = math.exp %92 : f16
                  linalg.yield %93 : f16
                } -> tensor<64x64x1xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %82 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%arg10 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in, %in_10 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x1xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<64x64x512xf16>) outs(%83 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %92 = arith.addf %in, %out : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %82 : tensor<64x64x128xf16>, tensor<64x64x1xf16>) outs(%42 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.mulf %in, %in_10 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %50 : memref<64x64x1xf16>
                %c0_8 = arith.constant 0 : index
                %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %70, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%24, %c0], LR : [%24, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
                %87 = loom.bufferize_to_tensor %61[64, 512, 128] : memref<64x512x128xf16> -> tensor<64x512x128xf16>
                %88 = linalg.fill ins(%cst : f16) outs(%45 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                %89 = linalg.batch_matmul ins(%81, %87 : tensor<64x64x512xf16>, tensor<64x512x128xf16>) outs(%88 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                loom.semaphore_give %61 : memref<64x512x128xf16>
                loom.semaphore_give %55 : memref<64x64x512xf16>
                %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %85 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%arg11 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %92 = arith.addf %in, %in_10 : f16
                  linalg.yield %92 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %44 : memref<64x64x128xf16>
                loom.semaphore_give %41 : memref<64x64x128xf16>
                %91 = linalg.copy ins(%78 : tensor<64x64x1xf16>) outs(%arg9 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                loom.semaphore_give %47 : memref<64x64x1xf16>
                scf.yield %91, %84, %90 : tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>
              }
              loom.semaphore_give %35 : memref<64x64x1xf16>
              loom.semaphore_give %22 : memref<64x64x128xf16>
              %63 = loom.broadcast ins(%62#1 : tensor<64x64x1xf16>) outs(%40 : tensor<64x64x128xf16>) dim(2) -> tensor<64x64x128xf16>
              loom.semaphore_give %31 : memref<64x64x1xf16>
              %64 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %65 = loom.semaphore_take %64 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %66 = loom.init_tensor %65[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62#2, %63 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%66 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %70 = arith.divf %in, %in_6 : f16
                linalg.yield %70 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %39 : memref<64x64x128xf16>
              loom.semaphore_give %27 : memref<64x64x128xf16>
              %c0_4 = arith.constant 0 : index
              %68 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%68], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %69 = loom.bufferize_to_memref %67 : tensor<64x64x128xf16> -> memref<64x64x128xf16>
              loom.copy %69, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%24, %arg6], LR : [%24, %arg6]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %65 : memref<64x64x128xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x2x4_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc2_dim_y_level0_bc8_dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (4) {
        affine.parallel (%arg5) = (0) to (2) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg5, %arg6, %arg7)
              %19 = arith.muli %arg4, %c64 : index
              %20 = arith.muli %18, %c64 : index
              %21 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %22 = loom.semaphore_take %21 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.muli %arg4, %c2 : index
              %25 = arith.addi %arg5, %24 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
              %26 = loom.bufferize_to_tensor %22[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %27 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %28 = loom.semaphore_take %27 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %29 = loom.init_tensor %28[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              %31 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %32 = loom.semaphore_take %31 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %33 = loom.init_tensor %32[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %34 = linalg.fill ins(%cst_0 : f16) outs(%33 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %35 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %36 = loom.semaphore_take %35 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %37 = loom.init_tensor %36[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %38 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %39 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %40 = loom.semaphore_take %39 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %41 = loom.init_tensor %40[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %42 = loom.semaphore_take %39 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %43 = loom.init_tensor %42[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %44 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %45 = loom.semaphore_take %44 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %46 = loom.init_tensor %45[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %47 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %48 = loom.semaphore_take %47 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %49 = loom.init_tensor %48[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %50 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %51 = loom.semaphore_take %50 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %52 = loom.init_tensor %51[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %53 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
              %54 = loom.semaphore_take %53 : memref<64x128x512xf16> -> memref<64x128x512xf16>
              %55 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %56 = loom.semaphore_take %55 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %57 = loom.init_tensor %56[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %58 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %59 = loom.semaphore_take %58 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %60 = loom.init_tensor %59[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %61 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
              %62 = loom.semaphore_take %61 : memref<64x512x128xf16> -> memref<64x512x128xf16>
              %63:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %38, %arg10 = %34, %arg11 = %30) -> (tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>) {
                %71 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%19, %c0_6, %71)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                %73 = arith.addi %24, %c1 : index
                loom.copy %reinterpret_cast_7, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%24, %c0], LR : [%73, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
                %74 = loom.bufferize_to_tensor %54[64, 128, 512] : memref<64x128x512xf16> -> tensor<64x128x512xf16>
                %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                %76 = linalg.batch_matmul ins(%26, %74 : tensor<64x64x128xf16>, tensor<64x128x512xf16>) outs(%75 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                loom.semaphore_give %54 : memref<64x128x512xf16>
                %77 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<64x64x512xf16>) outs(%77 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.maximumf %in, %out : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78 : tensor<64x64x1xf16>) outs(%49 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %79 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%49 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.cmpf ogt, %in, %in_10 : f16
                  %95 = arith.select %94, %in, %in_10 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<64x64x512xf16>) outs(%57 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x512xf16>
                %82 = loom.broadcast ins(%80 : tensor<64x64x1xf16>) outs(%60 : tensor<64x64x512xf16>) dim(2) -> tensor<64x64x512xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %82 : tensor<64x64x512xf16>, tensor<64x64x512xf16>) outs(%57 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.subf %in, %in_10 : f16
                  %95 = math.exp %94 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x512xf16>
                loom.semaphore_give %59 : memref<64x64x512xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %80 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%52 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.subf %in, %in_10 : f16
                  %95 = math.exp %94 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%arg10 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.mulf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%83 : tensor<64x64x512xf16>) outs(%85 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.addf %in, %out : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %84 : tensor<64x64x128xf16>, tensor<64x64x1xf16>) outs(%43 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.mulf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %51 : memref<64x64x1xf16>
                %c0_8 = arith.constant 0 : index
                %88 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %71, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%88], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%24, %c0], LR : [%73, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
                %89 = loom.bufferize_to_tensor %62[64, 512, 128] : memref<64x512x128xf16> -> tensor<64x512x128xf16>
                %90 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                %91 = linalg.batch_matmul ins(%83, %89 : tensor<64x64x512xf16>, tensor<64x512x128xf16>) outs(%90 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                loom.semaphore_give %62 : memref<64x512x128xf16>
                loom.semaphore_give %56 : memref<64x64x512xf16>
                %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %87 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%arg11 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.addf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %45 : memref<64x64x128xf16>
                loom.semaphore_give %42 : memref<64x64x128xf16>
                %93 = linalg.copy ins(%80 : tensor<64x64x1xf16>) outs(%arg9 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                loom.semaphore_give %48 : memref<64x64x1xf16>
                scf.yield %93, %86, %92 : tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>
              }
              loom.semaphore_give %36 : memref<64x64x1xf16>
              loom.semaphore_give %22 : memref<64x64x128xf16>
              %64 = loom.broadcast ins(%63#1 : tensor<64x64x1xf16>) outs(%41 : tensor<64x64x128xf16>) dim(2) -> tensor<64x64x128xf16>
              loom.semaphore_give %32 : memref<64x64x1xf16>
              %65 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %66 = loom.semaphore_take %65 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %67 = loom.init_tensor %66[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63#2, %64 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%67 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %71 = arith.divf %in, %in_6 : f16
                linalg.yield %71 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %40 : memref<64x64x128xf16>
              loom.semaphore_give %28 : memref<64x64x128xf16>
              %c0_4 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %70 = loom.bufferize_to_memref %68 : tensor<64x64x128xf16> -> memref<64x64x128xf16>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %66 : memref<64x64x128xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x4x2_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc4_dim_y_level0_bc8_dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (2) {
        affine.parallel (%arg5) = (0) to (4) {
          affine.parallel (%arg6) = (0) to (8) {
            scf.for %arg7 = %c0 to %c2 step %c1 {
              %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg5, %arg6, %arg7)
              %19 = arith.muli %arg4, %c64 : index
              %20 = arith.muli %18, %c64 : index
              %21 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %22 = loom.semaphore_take %21 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %c0_3 = arith.constant 0 : index
              %23 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_3)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %24 = arith.muli %arg4, %c4 : index
              %25 = arith.addi %arg5, %24 : index
              loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
              %26 = loom.bufferize_to_tensor %22[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %27 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %28 = loom.semaphore_take %27 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %29 = loom.init_tensor %28[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              %31 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %32 = loom.semaphore_take %31 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %33 = loom.init_tensor %32[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %34 = linalg.fill ins(%cst_0 : f16) outs(%33 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %35 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %36 = loom.semaphore_take %35 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %37 = loom.init_tensor %36[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %38 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %39 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %40 = loom.semaphore_take %39 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %41 = loom.init_tensor %40[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %42 = loom.semaphore_take %39 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %43 = loom.init_tensor %42[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %44 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %45 = loom.semaphore_take %44 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %46 = loom.init_tensor %45[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %47 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %48 = loom.semaphore_take %47 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %49 = loom.init_tensor %48[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %50 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
              %51 = loom.semaphore_take %50 : memref<64x64x1xf16> -> memref<64x64x1xf16>
              %52 = loom.init_tensor %51[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
              %53 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
              %54 = loom.semaphore_take %53 : memref<64x128x512xf16> -> memref<64x128x512xf16>
              %55 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %56 = loom.semaphore_take %55 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %57 = loom.init_tensor %56[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %58 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
              %59 = loom.semaphore_take %58 : memref<64x64x512xf16> -> memref<64x64x512xf16>
              %60 = loom.init_tensor %59[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
              %61 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
              %62 = loom.semaphore_take %61 : memref<64x512x128xf16> -> memref<64x512x128xf16>
              %63:3 = scf.for %arg8 = %c0 to %c8 step %c1 iter_args(%arg9 = %38, %arg10 = %34, %arg11 = %30) -> (tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>) {
                %71 = arith.muli %arg8, %c512 : index
                %c0_6 = arith.constant 0 : index
                %72 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%19, %c0_6, %71)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%72], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
                %73 = arith.addi %24, %c3 : index
                loom.copy %reinterpret_cast_7, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%24, %c0], LR : [%73, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
                %74 = loom.bufferize_to_tensor %54[64, 128, 512] : memref<64x128x512xf16> -> tensor<64x128x512xf16>
                %75 = linalg.fill ins(%cst : f16) outs(%57 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                %76 = linalg.batch_matmul ins(%26, %74 : tensor<64x64x128xf16>, tensor<64x128x512xf16>) outs(%75 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
                loom.semaphore_give %54 : memref<64x128x512xf16>
                %77 = linalg.fill ins(%cst_1 : f16) outs(%49 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%76 : tensor<64x64x512xf16>) outs(%77 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.maximumf %in, %out : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%78 : tensor<64x64x1xf16>) outs(%49 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %79 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%49 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.cmpf ogt, %in, %in_10 : f16
                  %95 = arith.select %94, %in, %in_10 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x1xf16>
                %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<64x64x512xf16>) outs(%57 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.mulf %in, %cst_2 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x512xf16>
                %82 = loom.broadcast ins(%80 : tensor<64x64x1xf16>) outs(%60 : tensor<64x64x512xf16>) dim(2) -> tensor<64x64x512xf16>
                %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %82 : tensor<64x64x512xf16>, tensor<64x64x512xf16>) outs(%57 : tensor<64x64x512xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.subf %in, %in_10 : f16
                  %95 = math.exp %94 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x512xf16>
                loom.semaphore_give %59 : memref<64x64x512xf16>
                %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %80 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%52 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.subf %in, %in_10 : f16
                  %95 = math.exp %94 : f16
                  linalg.yield %95 : f16
                } -> tensor<64x64x1xf16>
                %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %84 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%arg10 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.mulf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%83 : tensor<64x64x512xf16>) outs(%85 : tensor<64x64x1xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %94 = arith.addf %in, %out : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x1xf16>
                %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg11, %84 : tensor<64x64x128xf16>, tensor<64x64x1xf16>) outs(%43 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.mulf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %51 : memref<64x64x1xf16>
                %c0_8 = arith.constant 0 : index
                %88 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %71, %c0_8)
                %reinterpret_cast_9 = memref.reinterpret_cast %arg1 to offset: [%88], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
                loom.copy %reinterpret_cast_9, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%24, %c0], LR : [%73, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
                %89 = loom.bufferize_to_tensor %62[64, 512, 128] : memref<64x512x128xf16> -> tensor<64x512x128xf16>
                %90 = linalg.fill ins(%cst : f16) outs(%46 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                %91 = linalg.batch_matmul ins(%83, %89 : tensor<64x64x512xf16>, tensor<64x512x128xf16>) outs(%90 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
                loom.semaphore_give %62 : memref<64x512x128xf16>
                loom.semaphore_give %56 : memref<64x64x512xf16>
                %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%91, %87 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%arg11 : tensor<64x64x128xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %94 = arith.addf %in, %in_10 : f16
                  linalg.yield %94 : f16
                } -> tensor<64x64x128xf16>
                loom.semaphore_give %45 : memref<64x64x128xf16>
                loom.semaphore_give %42 : memref<64x64x128xf16>
                %93 = linalg.copy ins(%80 : tensor<64x64x1xf16>) outs(%arg9 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
                loom.semaphore_give %48 : memref<64x64x1xf16>
                scf.yield %93, %86, %92 : tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>
              }
              loom.semaphore_give %36 : memref<64x64x1xf16>
              loom.semaphore_give %22 : memref<64x64x128xf16>
              %64 = loom.broadcast ins(%63#1 : tensor<64x64x1xf16>) outs(%41 : tensor<64x64x128xf16>) dim(2) -> tensor<64x64x128xf16>
              loom.semaphore_give %32 : memref<64x64x1xf16>
              %65 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
              %66 = loom.semaphore_take %65 : memref<64x64x128xf16> -> memref<64x64x128xf16>
              %67 = loom.init_tensor %66[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
              %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63#2, %64 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%67 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_6: f16, %out: f16):
                %71 = arith.divf %in, %in_6 : f16
                linalg.yield %71 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %40 : memref<64x64x128xf16>
              loom.semaphore_give %28 : memref<64x64x128xf16>
              %c0_4 = arith.constant 0 : index
              %69 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%19, %20, %c0_4)
              %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%69], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              %70 = loom.bufferize_to_memref %68 : tensor<64x64x128xf16> -> memref<64x64x128xf16>
              loom.copy %70, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%25, %arg6], LR : [%25, %arg6]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.semaphore_give %66 : memref<64x64x128xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8x1_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level1_bc8_dim_y_level0_bc8_dim_x_level1_bc8_dim_y_level0_bc8_n__tile_b64__tile_m64__tile_n512(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg4) = (0) to (1) {
        affine.parallel (%arg5) = (0) to (8) {
          affine.parallel (%arg6) = (0) to (8) {
            %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg5, %arg6)
            %19 = arith.muli %18, %c64 : index
            %20 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %21 = loom.semaphore_take %20 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %22 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%c0_3, %19, %c0_4)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            %23 = arith.muli %arg4, %c8 : index
            %24 = arith.addi %arg5, %23 : index
            loom.copy %reinterpret_cast, %21 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%24, %arg6], LR : [%24, %arg6]) : memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x64x128xf16>
            %25 = loom.bufferize_to_tensor %21[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %26 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %28 = loom.init_tensor %27[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
            %30 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
            %31 = loom.semaphore_take %30 : memref<64x64x1xf16> -> memref<64x64x1xf16>
            %32 = loom.init_tensor %31[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
            %33 = linalg.fill ins(%cst_0 : f16) outs(%32 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
            %34 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
            %35 = loom.semaphore_take %34 : memref<64x64x1xf16> -> memref<64x64x1xf16>
            %36 = loom.init_tensor %35[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
            %37 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
            %38 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %39 = loom.semaphore_take %38 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %40 = loom.init_tensor %39[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %41 = loom.semaphore_take %38 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %42 = loom.init_tensor %41[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %43 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %44 = loom.semaphore_take %43 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %45 = loom.init_tensor %44[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %46 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
            %47 = loom.semaphore_take %46 : memref<64x64x1xf16> -> memref<64x64x1xf16>
            %48 = loom.init_tensor %47[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
            %49 = loom.alloc [64, 64, 1] on @L1 : memref<64x64x1xf16>
            %50 = loom.semaphore_take %49 : memref<64x64x1xf16> -> memref<64x64x1xf16>
            %51 = loom.init_tensor %50[64, 64, 1] : memref<64x64x1xf16> -> tensor<64x64x1xf16>
            %52 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
            %53 = loom.semaphore_take %52 : memref<64x128x512xf16> -> memref<64x128x512xf16>
            %54 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
            %55 = loom.semaphore_take %54 : memref<64x64x512xf16> -> memref<64x64x512xf16>
            %56 = loom.init_tensor %55[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
            %57 = loom.alloc [64, 64, 512] on @L1 : memref<64x64x512xf16>
            %58 = loom.semaphore_take %57 : memref<64x64x512xf16> -> memref<64x64x512xf16>
            %59 = loom.init_tensor %58[64, 64, 512] : memref<64x64x512xf16> -> tensor<64x64x512xf16>
            %60 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
            %61 = loom.semaphore_take %60 : memref<64x512x128xf16> -> memref<64x512x128xf16>
            %62:3 = scf.for %arg7 = %c0 to %c8 step %c1 iter_args(%arg8 = %37, %arg9 = %33, %arg10 = %29) -> (tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>) {
              %70 = arith.muli %arg7, %c512 : index
              %c0_8 = arith.constant 0 : index
              %c0_9 = arith.constant 0 : index
              %71 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 4096 + d2)>(%c0_8, %c0_9, %70)
              %reinterpret_cast_10 = memref.reinterpret_cast %arg0 to offset: [%71], sizes: [64, 128, 512], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_10, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x128x512xf16, strided<[524288, 4096, 1], offset: ?>> to memref<64x128x512xf16>
              %72 = loom.bufferize_to_tensor %53[64, 128, 512] : memref<64x128x512xf16> -> tensor<64x128x512xf16>
              %73 = linalg.fill ins(%cst : f16) outs(%56 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
              %74 = linalg.batch_matmul ins(%25, %72 : tensor<64x64x128xf16>, tensor<64x128x512xf16>) outs(%73 : tensor<64x64x512xf16>) -> tensor<64x64x512xf16>
              loom.semaphore_give %53 : memref<64x128x512xf16>
              %75 = linalg.fill ins(%cst_1 : f16) outs(%48 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<64x64x512xf16>) outs(%75 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %92 = arith.maximumf %in, %out : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x1xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%76 : tensor<64x64x1xf16>) outs(%48 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %92 = arith.mulf %in, %cst_2 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x1xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %77 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%48 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.cmpf ogt, %in, %in_14 : f16
                %93 = arith.select %92, %in, %in_14 : f16
                linalg.yield %93 : f16
              } -> tensor<64x64x1xf16>
              %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74 : tensor<64x64x512xf16>) outs(%56 : tensor<64x64x512xf16>) {
              ^bb0(%in: f16, %out: f16):
                %92 = arith.mulf %in, %cst_2 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x512xf16>
              %80 = loom.broadcast ins(%78 : tensor<64x64x1xf16>) outs(%59 : tensor<64x64x512xf16>) dim(2) -> tensor<64x64x512xf16>
              %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%79, %80 : tensor<64x64x512xf16>, tensor<64x64x512xf16>) outs(%56 : tensor<64x64x512xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.subf %in, %in_14 : f16
                %93 = math.exp %92 : f16
                linalg.yield %93 : f16
              } -> tensor<64x64x512xf16>
              loom.semaphore_give %58 : memref<64x64x512xf16>
              %82 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %78 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%51 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.subf %in, %in_14 : f16
                %93 = math.exp %92 : f16
                linalg.yield %93 : f16
              } -> tensor<64x64x1xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %82 : tensor<64x64x1xf16>, tensor<64x64x1xf16>) outs(%arg9 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.mulf %in, %in_14 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x1xf16>
              %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<64x64x512xf16>) outs(%83 : tensor<64x64x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %92 = arith.addf %in, %out : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x1xf16>
              %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg10, %82 : tensor<64x64x128xf16>, tensor<64x64x1xf16>) outs(%42 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.mulf %in, %in_14 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %50 : memref<64x64x1xf16>
              %c0_11 = arith.constant 0 : index
              %c0_12 = arith.constant 0 : index
              %86 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%c0_11, %70, %c0_12)
              %reinterpret_cast_13 = memref.reinterpret_cast %arg1 to offset: [%86], sizes: [64, 512, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_13, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x512x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<64x512x128xf16>
              %87 = loom.bufferize_to_tensor %61[64, 512, 128] : memref<64x512x128xf16> -> tensor<64x512x128xf16>
              %88 = linalg.fill ins(%cst : f16) outs(%45 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              %89 = linalg.batch_matmul ins(%81, %87 : tensor<64x64x512xf16>, tensor<64x512x128xf16>) outs(%88 : tensor<64x64x128xf16>) -> tensor<64x64x128xf16>
              loom.semaphore_give %61 : memref<64x512x128xf16>
              loom.semaphore_give %55 : memref<64x64x512xf16>
              %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%89, %85 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%arg10 : tensor<64x64x128xf16>) {
              ^bb0(%in: f16, %in_14: f16, %out: f16):
                %92 = arith.addf %in, %in_14 : f16
                linalg.yield %92 : f16
              } -> tensor<64x64x128xf16>
              loom.semaphore_give %44 : memref<64x64x128xf16>
              loom.semaphore_give %41 : memref<64x64x128xf16>
              %91 = linalg.copy ins(%78 : tensor<64x64x1xf16>) outs(%arg8 : tensor<64x64x1xf16>) -> tensor<64x64x1xf16>
              loom.semaphore_give %47 : memref<64x64x1xf16>
              scf.yield %91, %84, %90 : tensor<64x64x1xf16>, tensor<64x64x1xf16>, tensor<64x64x128xf16>
            }
            loom.semaphore_give %35 : memref<64x64x1xf16>
            loom.semaphore_give %21 : memref<64x64x128xf16>
            %63 = loom.broadcast ins(%62#1 : tensor<64x64x1xf16>) outs(%40 : tensor<64x64x128xf16>) dim(2) -> tensor<64x64x128xf16>
            loom.semaphore_give %31 : memref<64x64x1xf16>
            %64 = loom.alloc [64, 64, 128] on @L1 : memref<64x64x128xf16>
            %65 = loom.semaphore_take %64 : memref<64x64x128xf16> -> memref<64x64x128xf16>
            %66 = loom.init_tensor %65[64, 64, 128] : memref<64x64x128xf16> -> tensor<64x64x128xf16>
            %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%62#2, %63 : tensor<64x64x128xf16>, tensor<64x64x128xf16>) outs(%66 : tensor<64x64x128xf16>) {
            ^bb0(%in: f16, %in_8: f16, %out: f16):
              %70 = arith.divf %in, %in_8 : f16
              linalg.yield %70 : f16
            } -> tensor<64x64x128xf16>
            loom.semaphore_give %39 : memref<64x64x128xf16>
            loom.semaphore_give %27 : memref<64x64x128xf16>
            %c0_5 = arith.constant 0 : index
            %c0_6 = arith.constant 0 : index
            %68 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 524288 + d1 * 128 + d2)>(%c0_5, %19, %c0_6)
            %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%68], sizes: [64, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            %69 = loom.bufferize_to_memref %67 : tensor<64x64x128xf16> -> memref<64x64x128xf16>
            loom.copy %69, %reinterpret_cast_7 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%24, %arg6], LR : [%24, %arg6]) : memref<64x64x128xf16> to memref<64x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.semaphore_give %65 : memref<64x64x128xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
