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
            %33 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %34 = loom.init_tensor %33[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %35 = loom.semaphore_take %30 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %36 = loom.init_tensor %35[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %37 = linalg.fill ins(%cst : f16) outs(%36 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
            %38 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %39 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %40 = loom.init_tensor %39[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %41 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %42 = loom.init_tensor %41[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %43 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %44 = loom.init_tensor %43[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %45 = loom.semaphore_take %38 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %46 = loom.init_tensor %45[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %47 = linalg.fill ins(%cst_0 : f16) outs(%46 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
            %48 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %49 = loom.semaphore_take %48 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %50 = loom.init_tensor %49[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %51 = loom.semaphore_take %48 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %52 = loom.init_tensor %51[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %53 = linalg.fill ins(%cst_1 : f16) outs(%52 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
            %54 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %55 = loom.semaphore_take %54 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %56 = loom.init_tensor %55[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %57 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %58 = loom.semaphore_take %57 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %59 = loom.init_tensor %58[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %60 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %61 = loom.semaphore_take %60 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %62 = loom.init_tensor %61[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %63 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %64 = loom.semaphore_take %63 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %65 = loom.init_tensor %64[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %66 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
            %67 = loom.semaphore_take %66 : memref<1x128x64xf16> -> memref<1x128x64xf16>
            %68 = loom.alloc [1, 32, 64] on @L1 : memref<1x32x64xf16>
            %69 = loom.semaphore_take %68 : memref<1x32x64xf16> -> memref<1x32x64xf16>
            %70 = loom.init_tensor %69[1, 32, 64] : memref<1x32x64xf16> -> tensor<1x32x64xf16>
            %71 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
            %72 = loom.semaphore_take %71 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %73 = loom.init_tensor %72[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
            %74 = loom.semaphore_take %71 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %75 = loom.init_tensor %74[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
            %76 = loom.semaphore_take %71 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %77 = loom.init_tensor %76[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
            %78 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
            %79 = loom.semaphore_take %78 : memref<1x64x128xf16> -> memref<1x64x128xf16>
            %80:3 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %53, %arg9 = %47, %arg10 = %37) -> (tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x128xf16>) {
              %104 = arith.muli %arg7, %c64 : index
              %105 = arith.addi %29, %104 : index
              %c0_5 = arith.constant 0 : index
              %106 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %105)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%106], sizes: [1, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x64xf16>
              %107 = loom.bufferize_to_tensor %67[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %108 = linalg.fill ins(%cst : f16) outs(%70 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              %109 = linalg.batch_matmul ins(%28, %107 : tensor<1x32x128xf16>, tensor<1x128x64xf16>) outs(%108 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              loom.semaphore_give %67 : memref<1x128x64xf16>
              %110 = linalg.fill ins(%cst_1 : f16) outs(%59 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%109 : tensor<1x32x64xf16>) outs(%110 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %126 = arith.maximumf %in, %out : f16
                linalg.yield %126 : f16
              } -> tensor<1x32x1xf16>
              %112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %111 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%59 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %126 = arith.mulf %in_9, %cst_2 : f16
                %127 = arith.cmpf ogt, %in, %126 : f16
                %128 = arith.select %127, %in, %126 : f16
                linalg.yield %128 : f16
              } -> tensor<1x32x1xf16>
              %113 = loom.broadcast ins(%112 : tensor<1x32x1xf16>) outs(%77 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x64xf16>
              %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%109, %113 : tensor<1x32x64xf16>, tensor<1x32x64xf16>) outs(%70 : tensor<1x32x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %126 = arith.mulf %in, %cst_2 : f16
                %127 = arith.subf %126, %in_9 : f16
                %128 = math.exp %127 : f16
                linalg.yield %128 : f16
              } -> tensor<1x32x64xf16>
              loom.semaphore_give %76 : memref<1x32x32xf16>
              %115 = linalg.fill ins(%cst : f16) outs(%62 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %116 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%114 : tensor<1x32x64xf16>) outs(%115 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %126 = arith.addf %in, %out : f16
                linalg.yield %126 : f16
              } -> tensor<1x32x1xf16>
              %117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %112 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%65 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %126 = arith.subf %in, %in_9 : f16
                %127 = math.exp %126 : f16
                linalg.yield %127 : f16
              } -> tensor<1x32x1xf16>
              %118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %117, %116 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%arg9 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %126 = arith.mulf %in, %in_9 : f16
                %127 = arith.addf %126, %in_10 : f16
                linalg.yield %127 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %61 : memref<1x32x1xf16>
              %119 = loom.broadcast ins(%117 : tensor<1x32x1xf16>) outs(%75 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %64 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %120 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %105, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%120], sizes: [1, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x64x128xf16>
              %121 = loom.bufferize_to_tensor %79[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %122 = linalg.fill ins(%cst : f16) outs(%56 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %123 = linalg.batch_matmul ins(%114, %121 : tensor<1x32x64xf16>, tensor<1x64x128xf16>) outs(%122 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %79 : memref<1x64x128xf16>
              loom.semaphore_give %69 : memref<1x32x64xf16>
              %124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%123, %arg10, %119 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%arg10 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %126 = arith.mulf %in_9, %in_10 : f16
                %127 = arith.addf %in, %126 : f16
                linalg.yield %127 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %74 : memref<1x32x32xf16>
              loom.semaphore_give %55 : memref<1x32x128xf16>
              %125 = linalg.copy ins(%112 : tensor<1x32x1xf16>) outs(%arg8 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %58 : memref<1x32x1xf16>
              scf.yield %125, %118, %124 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x128xf16>
            }
            loom.semaphore_give %26 : memref<1x32x128xf16>
            %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80#1, %80#0 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%50 : tensor<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %104 = math.log %in : f16
              %105 = arith.addf %104, %in_5 : f16
              linalg.yield %105 : f16
            } -> tensor<1x32x1xf16>
            loom.semaphore_give %51 : memref<1x32x1xf16>
            %82 = loom.broadcast ins(%80#1 : tensor<1x32x1xf16>) outs(%73 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
            loom.semaphore_give %45 : memref<1x32x1xf16>
            %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%80#2, %82 : tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%25 : tensor<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %104 = arith.divf %in, %in_5 : f16
              linalg.yield %104 : f16
            } -> tensor<1x32x128xf16>
            loom.semaphore_give %72 : memref<1x32x32xf16>
            loom.semaphore_give %35 : memref<1x32x128xf16>
            %84 = loom.sync ins(%81 : tensor<1x32x1xf16>) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
            loom.semaphore_give %49 : memref<1x32x1xf16>
            %85 = loom.alloc [8, 1, 32, 1] on @L1 : memref<8x1x32x1xf16>
            %86 = loom.semaphore_take %85 : memref<8x1x32x1xf16> -> memref<8x1x32x1xf16>
            %87 = loom.init_tensor %86[8, 1, 32, 1] : memref<8x1x32x1xf16> -> tensor<8x1x32x1xf16>
            %88 = loom.gather ins(%84 : tensor<1x32x1xf16>) outs(%87 : tensor<8x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x1x32x1xf16>
            loom.semaphore_give %43 : memref<1x32x1xf16>
            %89 = loom.sync ins(%83 : tensor<1x32x128xf16>) outs(%34 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
            loom.semaphore_give %24 : memref<1x32x128xf16>
            %90 = loom.alloc [8, 1, 32, 128] on @L1 : memref<8x1x32x128xf16>
            %91 = loom.semaphore_take %90 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
            %92 = loom.init_tensor %91[8, 1, 32, 128] : memref<8x1x32x128xf16> -> tensor<8x1x32x128xf16>
            %93 = loom.gather ins(%89 : tensor<1x32x128xf16>) outs(%92 : tensor<8x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x1x32x128xf16>
            loom.semaphore_give %33 : memref<1x32x128xf16>
            %94 = loom.alloc [8, 1, 32, 1] on @L1 : memref<8x1x32x1xf16>
            %95 = loom.semaphore_take %94 : memref<8x1x32x1xf16> -> memref<8x1x32x1xf16>
            %96 = loom.init_tensor %95[8, 1, 32, 1] : memref<8x1x32x1xf16> -> tensor<8x1x32x1xf16>
            %97 = loom.alloc [8, 1, 32, 128] on @L1 : memref<8x1x32x128xf16>
            %98 = loom.semaphore_take %97 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
            %99 = loom.init_tensor %98[8, 1, 32, 128] : memref<8x1x32x128xf16> -> tensor<8x1x32x128xf16>
            %100 = loom.alloc [8, 1, 32, 32] on @L1 : memref<8x1x32x32xf16>
            %101 = loom.semaphore_take %100 : memref<8x1x32x32xf16> -> memref<8x1x32x32xf16>
            %102 = loom.init_tensor %101[8, 1, 32, 32] : memref<8x1x32x32xf16> -> tensor<8x1x32x32xf16>
            %103 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %103 {
              %104 = linalg.fill ins(%cst_1 : f16) outs(%42 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %105 = loom.sync ins(%88 : tensor<8x1x32x1xf16>) outs(%96 : tensor<8x1x32x1xf16>) -> tensor<8x1x32x1xf16>
              loom.semaphore_give %86 : memref<8x1x32x1xf16>
              %106 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%105 : tensor<8x1x32x1xf16>) outs(%104 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %119 = arith.maximumf %in, %out : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%105, %106 : tensor<8x1x32x1xf16>, tensor<1x32x1xf16>) outs(%96 : tensor<8x1x32x1xf16>) {
              ^bb0(%in: f16, %in_8: f16, %out: f16):
                %119 = arith.subf %in, %in_8 : f16
                %120 = math.exp %119 : f16
                linalg.yield %120 : f16
              } -> tensor<8x1x32x1xf16>
              loom.semaphore_give %41 : memref<1x32x1xf16>
              %108 = linalg.fill ins(%cst : f16) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %109 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%107 : tensor<8x1x32x1xf16>) outs(%108 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %119 = arith.addf %in, %out : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x1xf16>
              %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%107, %109 : tensor<8x1x32x1xf16>, tensor<1x32x1xf16>) outs(%96 : tensor<8x1x32x1xf16>) {
              ^bb0(%in: f16, %in_8: f16, %out: f16):
                %119 = arith.divf %in, %in_8 : f16
                linalg.yield %119 : f16
              } -> tensor<8x1x32x1xf16>
              loom.semaphore_give %39 : memref<1x32x1xf16>
              %111 = loom.broadcast ins(%110 : tensor<8x1x32x1xf16>) outs(%102 : tensor<8x1x32x32xf16>) dim(3) -> tensor<8x1x32x128xf16>
              loom.semaphore_give %95 : memref<8x1x32x1xf16>
              %112 = loom.sync ins(%93 : tensor<8x1x32x128xf16>) outs(%99 : tensor<8x1x32x128xf16>) -> tensor<8x1x32x128xf16>
              loom.semaphore_give %91 : memref<8x1x32x128xf16>
              %113 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%112, %111 : tensor<8x1x32x128xf16>, tensor<8x1x32x128xf16>) outs(%99 : tensor<8x1x32x128xf16>) {
              ^bb0(%in: f16, %in_8: f16, %out: f16):
                %119 = arith.mulf %in, %in_8 : f16
                linalg.yield %119 : f16
              } -> tensor<8x1x32x128xf16>
              loom.semaphore_give %101 : memref<8x1x32x32xf16>
              %114 = linalg.fill ins(%cst : f16) outs(%23 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %115 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%113 : tensor<8x1x32x128xf16>) outs(%114 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %119 = arith.addf %in, %out : f16
                linalg.yield %119 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %98 : memref<8x1x32x128xf16>
              %116 = loom.sync ins(%115 : tensor<1x32x128xf16>) outs(%32 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %22 : memref<1x32x128xf16>
              %c0_5 = arith.constant 0 : index
              %c0_6 = arith.constant 0 : index
              %117 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_5, %c0_6)
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%117], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %118 = loom.bufferize_to_memref %116 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              loom.copy %118, %reinterpret_cast_7 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.semaphore_give %31 : memref<1x32x128xf16>
            }
          } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
