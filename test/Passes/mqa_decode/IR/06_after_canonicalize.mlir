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
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
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
            %25 = loom.init_tensor %24[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %26 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %27 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_3, %c0_4)
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%27], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
            %28 = loom.bufferize_to_tensor %26[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %29 = arith.muli %arg5, %c1024 : index
            %30 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %31 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %32 = loom.init_tensor %31[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %33 = linalg.fill ins(%cst : f16) outs(%32 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
            %34 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %35 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %37 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %39 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %40 = loom.init_tensor %39[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
            %cast = tensor.cast %41 : tensor<1x32x1xf16> to tensor<?x32x1xf16>
            %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %44 = loom.init_tensor %43[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %45 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %46 = loom.init_tensor %45[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %47 = linalg.fill ins(%cst_1 : f16) outs(%46 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
            %48 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %49 = loom.semaphore_take %48 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %50 = loom.init_tensor %49[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %51 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %52 = loom.semaphore_take %51 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %53 = loom.init_tensor %52[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %54 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %55 = loom.semaphore_take %54 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %56 = loom.init_tensor %55[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %57 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %58 = loom.semaphore_take %57 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %59 = loom.init_tensor %58[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %60 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
            %61 = loom.semaphore_take %60 : memref<1x128x64xf16> -> memref<1x128x64xf16>
            %62 = loom.alloc [1, 32, 64] on @L1 : memref<1x32x64xf16>
            %63 = loom.semaphore_take %62 : memref<1x32x64xf16> -> memref<1x32x64xf16>
            %64 = loom.init_tensor %63[1, 32, 64] : memref<1x32x64xf16> -> tensor<1x32x64xf16>
            %65 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
            %66 = loom.semaphore_take %65 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %67 = loom.init_tensor %66[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
            %68 = loom.semaphore_take %65 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %69 = loom.init_tensor %68[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
            %70 = loom.semaphore_take %65 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %71 = loom.init_tensor %70[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
            %72 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
            %73 = loom.semaphore_take %72 : memref<1x64x128xf16> -> memref<1x64x128xf16>
            %74:3 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %47, %arg9 = %cast, %arg10 = %33) -> (tensor<1x32x1xf16>, tensor<?x32x1xf16>, tensor<1x32x128xf16>) {
              %96 = arith.muli %arg7, %c64 : index
              %97 = arith.addi %29, %96 : index
              %c0_6 = arith.constant 0 : index
              %98 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_6, %97)
              %reinterpret_cast_7 = memref.reinterpret_cast %arg0 to offset: [%98], sizes: [1, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
              loom.copy %reinterpret_cast_7, %61 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x64xf16>
              %99 = loom.bufferize_to_tensor %61[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %100 = linalg.fill ins(%cst : f16) outs(%64 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              %101 = linalg.batch_matmul ins(%28, %99 : tensor<1x32x128xf16>, tensor<1x128x64xf16>) outs(%100 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              loom.semaphore_give %61 : memref<1x128x64xf16>
              %102 = linalg.fill ins(%cst_1 : f16) outs(%53 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%101 : tensor<1x32x64xf16>) outs(%102 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.maximumf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %103 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%53 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_13: f16, %out: f16):
                %118 = arith.mulf %in_13, %cst_2 : f16
                %119 = arith.cmpf ogt, %in, %118 : f16
                %120 = arith.select %119, %in, %118 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x1xf16>
              %105 = loom.broadcast ins(%104 : tensor<1x32x1xf16>) outs(%71 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x64xf16>
              %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101, %105 : tensor<1x32x64xf16>, tensor<1x32x64xf16>) outs(%64 : tensor<1x32x64xf16>) {
              ^bb0(%in: f16, %in_13: f16, %out: f16):
                %118 = arith.mulf %in, %cst_2 : f16
                %119 = arith.subf %118, %in_13 : f16
                %120 = math.exp %119 : f16
                linalg.yield %120 : f16
              } -> tensor<1x32x64xf16>
              loom.semaphore_give %70 : memref<1x32x32xf16>
              %107 = linalg.fill ins(%cst : f16) outs(%56 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%106 : tensor<1x32x64xf16>) outs(%107 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %118 = arith.addf %in, %out : f16
                linalg.yield %118 : f16
              } -> tensor<1x32x1xf16>
              %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %104 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%59 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_13: f16, %out: f16):
                %118 = arith.subf %in, %in_13 : f16
                %119 = math.exp %118 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %cast_8 = tensor.cast %arg9 : tensor<?x32x1xf16> to tensor<1x32x1xf16>
              %cast_9 = tensor.cast %arg9 : tensor<?x32x1xf16> to tensor<1x32x1xf16>
              %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cast_8, %109, %108 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%cast_9 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_13: f16, %in_14: f16, %out: f16):
                %118 = arith.mulf %in, %in_13 : f16
                %119 = arith.addf %118, %in_14 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %cast_10 = tensor.cast %110 : tensor<1x32x1xf16> to tensor<?x32x1xf16>
              loom.semaphore_give %55 : memref<1x32x1xf16>
              %111 = loom.broadcast ins(%109 : tensor<1x32x1xf16>) outs(%69 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %58 : memref<1x32x1xf16>
              %c0_11 = arith.constant 0 : index
              %112 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %97, %c0_11)
              %reinterpret_cast_12 = memref.reinterpret_cast %arg1 to offset: [%112], sizes: [1, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_12, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x64x128xf16>
              %113 = loom.bufferize_to_tensor %73[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %114 = linalg.fill ins(%cst : f16) outs(%50 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %115 = linalg.batch_matmul ins(%106, %113 : tensor<1x32x64xf16>, tensor<1x64x128xf16>) outs(%114 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %73 : memref<1x64x128xf16>
              loom.semaphore_give %63 : memref<1x32x64xf16>
              %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%115, %arg10, %111 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%arg10 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_13: f16, %in_14: f16, %out: f16):
                %118 = arith.mulf %in_13, %in_14 : f16
                %119 = arith.addf %in, %118 : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %68 : memref<1x32x32xf16>
              loom.semaphore_give %49 : memref<1x32x128xf16>
              %117 = linalg.copy ins(%104 : tensor<1x32x1xf16>) outs(%arg8 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %52 : memref<1x32x1xf16>
              scf.yield %117, %cast_10, %116 : tensor<1x32x1xf16>, tensor<?x32x1xf16>, tensor<1x32x128xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %26 : memref<1x32x128xf16>
            %cast_5 = tensor.cast %74#1 : tensor<?x32x1xf16> to tensor<1x32x1xf16>
            %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cast_5, %74#0 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%44 : tensor<1x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %96 = math.log %in : f16
              %97 = arith.addf %96, %in_6 : f16
              linalg.yield %97 : f16
            } -> tensor<1x32x1xf16>
            loom.semaphore_give %45 : memref<1x32x1xf16>
            %76 = loom.broadcast ins(%74#1 : tensor<?x32x1xf16>) outs(%67 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
            loom.semaphore_give %39 : memref<1x32x1xf16>
            %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%74#2, %76 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%25 : tensor<1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %96 = arith.divf %in, %in_6 : f16
              linalg.yield %96 : f16
            } -> tensor<1x32x128xf16>
            loom.semaphore_give %66 : memref<1x32x32xf16>
            loom.semaphore_give %31 : memref<1x32x128xf16>
            %78 = loom.alloc [8, 1, 32, 1] on @L1 : memref<8x1x32x1xf16>
            %79 = loom.semaphore_take %78 : memref<8x1x32x1xf16> -> memref<8x1x32x1xf16>
            %80 = loom.init_tensor %79[8, 1, 32, 1] : memref<8x1x32x1xf16> -> tensor<8x1x32x1xf16>
            %81 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %82 = loom.init_tensor %81[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %83 = loom.sync ins(%75 : tensor<1x32x1xf16>) outs(%82 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
            %84 = loom.gather ins(%83 : tensor<1x32x1xf16>) outs(%80 : tensor<8x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x1x32x1xf16>
            loom.semaphore_give %81 : memref<1x32x1xf16>
            loom.semaphore_give %43 : memref<1x32x1xf16>
            %85 = loom.alloc [8, 1, 32, 128] on @L1 : memref<8x1x32x128xf16>
            %86 = loom.semaphore_take %85 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
            %87 = loom.init_tensor %86[8, 1, 32, 128] : memref<8x1x32x128xf16> -> tensor<8x1x32x128xf16>
            %88 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %89 = loom.init_tensor %88[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %90 = loom.sync ins(%77 : tensor<1x32x128xf16>) outs(%89 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
            %91 = loom.gather ins(%90 : tensor<1x32x128xf16>) outs(%87 : tensor<8x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x1x32x128xf16>
            loom.semaphore_give %88 : memref<1x32x128xf16>
            loom.semaphore_give %24 : memref<1x32x128xf16>
            %92 = loom.alloc [1024, 1, 32, 32] on @L1 : memref<1024x1x32x32xf16>
            %93 = loom.semaphore_take %92 : memref<1024x1x32x32xf16> -> memref<1024x1x32x32xf16>
            %94 = loom.init_tensor %93[1024, 1, 32, 32] : memref<1024x1x32x32xf16> -> tensor<1024x1x32x32xf16>
            %95 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %95 {
              %96 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %97 = loom.semaphore_take %78 : memref<8x1x32x1xf16> -> memref<8x1x32x1xf16>
              %98 = loom.init_tensor %97[8, 1, 32, 1] : memref<8x1x32x1xf16> -> tensor<8x1x32x1xf16>
              %99 = loom.sync ins(%84 : tensor<8x1x32x1xf16>) outs(%98 : tensor<8x1x32x1xf16>) -> tensor<8x1x32x1xf16>
              %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%99 : tensor<8x1x32x1xf16>) outs(%96 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.maximumf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %100 : tensor<8x1x32x1xf16>, tensor<1x32x1xf16>) outs(%80 : tensor<8x1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.subf %in, %in_9 : f16
                %118 = math.exp %117 : f16
                linalg.yield %118 : f16
              } -> tensor<8x1x32x1xf16>
              loom.semaphore_give %37 : memref<1x32x1xf16>
              %102 = linalg.fill ins(%cst : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%101 : tensor<8x1x32x1xf16>) outs(%102 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.addf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x1xf16>
              %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%101, %103 : tensor<8x1x32x1xf16>, tensor<1x32x1xf16>) outs(%80 : tensor<8x1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.divf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<8x1x32x1xf16>
              loom.semaphore_give %35 : memref<1x32x1xf16>
              %105 = loom.broadcast ins(%104 : tensor<8x1x32x1xf16>) outs(%94 : tensor<1024x1x32x32xf16>) dim(3) -> tensor<8x1x32x128xf16>
              loom.semaphore_give %97 : memref<8x1x32x1xf16>
              loom.semaphore_give %79 : memref<8x1x32x1xf16>
              %106 = loom.semaphore_take %85 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
              %107 = loom.init_tensor %106[8, 1, 32, 128] : memref<8x1x32x128xf16> -> tensor<8x1x32x128xf16>
              %108 = loom.sync ins(%91 : tensor<8x1x32x128xf16>) outs(%107 : tensor<8x1x32x128xf16>) -> tensor<8x1x32x128xf16>
              %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%108, %105 : tensor<8x1x32x128xf16>, tensor<8x1x32x128xf16>) outs(%87 : tensor<8x1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %117 = arith.mulf %in, %in_9 : f16
                linalg.yield %117 : f16
              } -> tensor<8x1x32x128xf16>
              loom.semaphore_give %93 : memref<1024x1x32x32xf16>
              %110 = linalg.fill ins(%cst : f16) outs(%23 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%109 : tensor<8x1x32x128xf16>) outs(%110 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %117 = arith.addf %in, %out : f16
                linalg.yield %117 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %106 : memref<8x1x32x128xf16>
              loom.semaphore_give %86 : memref<8x1x32x128xf16>
              %c0_6 = arith.constant 0 : index
              %c0_7 = arith.constant 0 : index
              %112 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_6, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg2 to offset: [%112], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %113 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %114 = loom.init_tensor %113[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %115 = loom.sync ins(%111 : tensor<1x32x128xf16>) outs(%114 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %116 = loom.bufferize_to_memref %115 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              loom.copy %116, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.semaphore_give %113 : memref<1x32x128xf16>
              loom.semaphore_give %22 : memref<1x32x128xf16>
            }
          } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
