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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_n__tile_b1__tile_n64__tile_s1024(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg4) = (0) to (8) {
        affine.parallel (%arg5) = (0) to (8) {
          scf.for %arg6 = %c0 to %c2 step %c1 {
            %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
            %21 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %22 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %23 = loom.init_tensor %22[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %24 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %c0_4 = arith.constant 0 : index
            %c0_5 = arith.constant 0 : index
            %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_4, %c0_5)
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
            %26 = loom.bufferize_to_tensor %24[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %27 = arith.muli %arg5, %c1024 : index
            %28 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %29 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %30 = loom.init_tensor %29[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %31 = linalg.fill ins(%cst_0 : f16) outs(%30 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
            %32 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %33 = loom.semaphore_take %32 : memref<1x32xf16> -> memref<1x32xf16>
            %34 = loom.init_tensor %33[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %35 = linalg.fill ins(%cst_1 : f16) outs(%34 : tensor<1x32xf16>) -> tensor<1x32xf16>
            %cast = tensor.cast %35 : tensor<1x32xf16> to tensor<?x32xf16>
            %36 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %37 = loom.semaphore_take %36 : memref<1x32xf16> -> memref<1x32xf16>
            %38 = loom.init_tensor %37[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %39 = linalg.fill ins(%cst_2 : f16) outs(%38 : tensor<1x32xf16>) -> tensor<1x32xf16>
            %cast_6 = tensor.cast %39 : tensor<1x32xf16> to tensor<?x32xf16>
            %40 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %41 = loom.semaphore_take %40 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %42 = loom.init_tensor %41[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %43 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %44 = loom.semaphore_take %43 : memref<1x32xf16> -> memref<1x32xf16>
            %45 = loom.init_tensor %44[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %46 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %47 = loom.semaphore_take %46 : memref<1x32xf16> -> memref<1x32xf16>
            %48 = loom.init_tensor %47[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %49 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
            %50 = loom.semaphore_take %49 : memref<1x32xf16> -> memref<1x32xf16>
            %51 = loom.init_tensor %50[1, 32] : memref<1x32xf16> -> tensor<1x32xf16>
            %52 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
            %53 = loom.semaphore_take %52 : memref<1x128x64xf16> -> memref<1x128x64xf16>
            %54 = loom.alloc [1, 32, 64] on @L1 : memref<1x32x64xf16>
            %55 = loom.semaphore_take %54 : memref<1x32x64xf16> -> memref<1x32x64xf16>
            %56 = loom.init_tensor %55[1, 32, 64] : memref<1x32x64xf16> -> tensor<1x32x64xf16>
            %57 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
            %58 = loom.semaphore_take %57 : memref<1x64x128xf16> -> memref<1x64x128xf16>
            %59:3 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %cast_6, %arg9 = %cast, %arg10 = %31) -> (tensor<?x32xf16>, tensor<?x32xf16>, tensor<1x32x128xf16>) {
              %65 = arith.muli %arg7, %c64 : index
              %66 = arith.addi %27, %65 : index
              %c0_7 = arith.constant 0 : index
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_7, %66)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg0 to offset: [%67], sizes: [1, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x64xf16>
              %68 = loom.bufferize_to_tensor %53[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %69 = linalg.fill ins(%cst_0 : f16) outs(%56 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              %70 = linalg.batch_matmul ins(%26, %68 : tensor<1x32x128xf16>, tensor<1x128x64xf16>) outs(%69 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              loom.semaphore_give %53 : memref<1x128x64xf16>
              %71 = linalg.fill ins(%cst_2 : f16) outs(%45 : tensor<1x32xf16>) -> tensor<1x32xf16>
              %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%70 : tensor<1x32x64xf16>) outs(%71 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %85 = arith.maximumf %in, %out : f16
                linalg.yield %85 : f16
              } -> tensor<1x32xf16>
              %cast_9 = tensor.cast %arg8 : tensor<?x32xf16> to tensor<1x32xf16>
              %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cast_9, %72 : tensor<1x32xf16>, tensor<1x32xf16>) outs(%45 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %in_18: f16, %out: f16):
                %85 = arith.mulf %in_18, %cst_3 : f16
                %86 = arith.cmpf ogt, %in, %85 : f16
                %87 = arith.select %86, %in, %85 : f16
                linalg.yield %87 : f16
              } -> tensor<1x32xf16>
              %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70, %73 : tensor<1x32x64xf16>, tensor<1x32xf16>) outs(%56 : tensor<1x32x64xf16>) {
              ^bb0(%in: f16, %in_18: f16, %out: f16):
                %85 = arith.mulf %in, %cst_3 : f16
                %86 = arith.subf %85, %in_18 : f16
                %87 = math.powf %cst, %86 : f16
                linalg.yield %87 : f16
              } -> tensor<1x32x64xf16>
              %75 = linalg.fill ins(%cst_0 : f16) outs(%48 : tensor<1x32xf16>) -> tensor<1x32xf16>
              %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%74 : tensor<1x32x64xf16>) outs(%75 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %out: f16):
                %85 = arith.addf %in, %out : f16
                linalg.yield %85 : f16
              } -> tensor<1x32xf16>
              %cast_10 = tensor.cast %arg8 : tensor<?x32xf16> to tensor<1x32xf16>
              %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cast_10, %73 : tensor<1x32xf16>, tensor<1x32xf16>) outs(%51 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %in_18: f16, %out: f16):
                %85 = arith.subf %in, %in_18 : f16
                %86 = math.powf %cst, %85 : f16
                linalg.yield %86 : f16
              } -> tensor<1x32xf16>
              %cast_11 = tensor.cast %arg9 : tensor<?x32xf16> to tensor<1x32xf16>
              %cast_12 = tensor.cast %arg9 : tensor<?x32xf16> to tensor<1x32xf16>
              %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cast_11, %77, %76 : tensor<1x32xf16>, tensor<1x32xf16>, tensor<1x32xf16>) outs(%cast_12 : tensor<1x32xf16>) {
              ^bb0(%in: f16, %in_18: f16, %in_19: f16, %out: f16):
                %85 = arith.mulf %in, %in_18 : f16
                %86 = arith.addf %85, %in_19 : f16
                linalg.yield %86 : f16
              } -> tensor<1x32xf16>
              %cast_13 = tensor.cast %78 : tensor<1x32xf16> to tensor<?x32xf16>
              loom.semaphore_give %47 : memref<1x32xf16>
              %c0_14 = arith.constant 0 : index
              %79 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %66, %c0_14)
              %reinterpret_cast_15 = memref.reinterpret_cast %arg1 to offset: [%79], sizes: [1, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_15, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x64x128xf16>
              %80 = loom.bufferize_to_tensor %58[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %81 = linalg.fill ins(%cst_0 : f16) outs(%42 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %82 = linalg.batch_matmul ins(%74, %80 : tensor<1x32x64xf16>, tensor<1x64x128xf16>) outs(%81 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %58 : memref<1x64x128xf16>
              loom.semaphore_give %55 : memref<1x32x64xf16>
              %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %arg10, %77 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32xf16>) outs(%arg10 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_18: f16, %in_19: f16, %out: f16):
                %85 = arith.mulf %in_18, %in_19 : f16
                %86 = arith.addf %in, %85 : f16
                linalg.yield %86 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %50 : memref<1x32xf16>
              loom.semaphore_give %41 : memref<1x32x128xf16>
              %cast_16 = tensor.cast %arg8 : tensor<?x32xf16> to tensor<1x32xf16>
              %84 = linalg.copy ins(%73 : tensor<1x32xf16>) outs(%cast_16 : tensor<1x32xf16>) -> tensor<1x32xf16>
              %cast_17 = tensor.cast %84 : tensor<1x32xf16> to tensor<?x32xf16>
              loom.semaphore_give %44 : memref<1x32xf16>
              scf.yield %cast_17, %cast_13, %83 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<1x32x128xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %33 : memref<1x32xf16>
            loom.semaphore_give %37 : memref<1x32xf16>
            loom.semaphore_give %24 : memref<1x32x128xf16>
            %60 = loom.alloc [8, 1, 32, 128] on @L1 : memref<8x1x32x128xf16>
            %61 = loom.semaphore_take %60 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
            %62 = loom.init_tensor %61[8, 1, 32, 128] : memref<8x1x32x128xf16> -> tensor<8x1x32x128xf16>
            %63 = loom.gather ins(%59#2 : tensor<1x32x128xf16>) outs(%62 : tensor<8x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x1x32x128xf16>
            loom.semaphore_give %29 : memref<1x32x128xf16>
            %64 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %64 {
              %65 = linalg.fill ins(%cst_0 : f16) outs(%23 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%63 : tensor<8x1x32x128xf16>) outs(%65 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %69 = arith.addf %in, %out : f16
                linalg.yield %69 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %61 : memref<8x1x32x128xf16>
              %c0_7 = arith.constant 0 : index
              %c0_8 = arith.constant 0 : index
              %67 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_7, %c0_8)
              %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%67], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %68 = loom.bufferize_to_memref %66 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              loom.copy %68, %reinterpret_cast_9 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.semaphore_give %22 : memref<1x32x128xf16>
            }
          } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
