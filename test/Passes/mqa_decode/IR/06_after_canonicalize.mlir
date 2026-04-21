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
            %c0_3 = arith.constant 0 : index
            %c0_4 = arith.constant 0 : index
            %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_3, %c0_4)
            %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
            %26 = loom.bufferize_to_tensor %24[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %27 = arith.muli %arg5, %c1024 : index
            %28 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %29 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %30 = loom.init_tensor %29[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
            %32 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %33 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %34 = loom.init_tensor %33[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %35 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %36 = loom.init_tensor %35[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %37 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %38 = loom.init_tensor %37[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %39 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %40 = loom.init_tensor %39[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %41 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
            %42 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %43 = loom.semaphore_take %42 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %44 = loom.init_tensor %43[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %45 = linalg.fill ins(%cst_1 : f16) outs(%44 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
            %46 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
            %47 = loom.semaphore_take %46 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %48 = loom.init_tensor %47[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %49 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %50 = loom.semaphore_take %49 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %51 = loom.init_tensor %50[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %52 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %53 = loom.semaphore_take %52 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %54 = loom.init_tensor %53[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %55 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
            %56 = loom.semaphore_take %55 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %57 = loom.init_tensor %56[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %58 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
            %59 = loom.semaphore_take %58 : memref<1x128x64xf16> -> memref<1x128x64xf16>
            %60 = loom.alloc [1, 32, 64] on @L1 : memref<1x32x64xf16>
            %61 = loom.semaphore_take %60 : memref<1x32x64xf16> -> memref<1x32x64xf16>
            %62 = loom.init_tensor %61[1, 32, 64] : memref<1x32x64xf16> -> tensor<1x32x64xf16>
            %63 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
            %64 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %65 = loom.init_tensor %64[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
            %66 = loom.semaphore_take %63 : memref<1x32x32xf16> -> memref<1x32x32xf16>
            %67 = loom.init_tensor %66[1, 32, 32] : memref<1x32x32xf16> -> tensor<1x32x32xf16>
            %68 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
            %69 = loom.semaphore_take %68 : memref<1x64x128xf16> -> memref<1x64x128xf16>
            %70:3 = scf.for %arg7 = %c0 to %c16 step %c1 iter_args(%arg8 = %45, %arg9 = %41, %arg10 = %31) -> (tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x128xf16>) {
              %90 = arith.muli %arg7, %c64 : index
              %91 = arith.addi %27, %90 : index
              %c0_5 = arith.constant 0 : index
              %92 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 8192 + d2)>(%20, %c0_5, %91)
              %reinterpret_cast_6 = memref.reinterpret_cast %arg0 to offset: [%92], sizes: [1, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
              loom.copy %reinterpret_cast_6, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x64xf16>
              %93 = loom.bufferize_to_tensor %59[1, 128, 64] : memref<1x128x64xf16> -> tensor<1x128x64xf16>
              %94 = linalg.fill ins(%cst : f16) outs(%62 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              %95 = linalg.batch_matmul ins(%26, %93 : tensor<1x32x128xf16>, tensor<1x128x64xf16>) outs(%94 : tensor<1x32x64xf16>) -> tensor<1x32x64xf16>
              loom.semaphore_give %59 : memref<1x128x64xf16>
              %96 = linalg.fill ins(%cst_1 : f16) outs(%51 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%95 : tensor<1x32x64xf16>) outs(%96 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %112 = arith.maximumf %in, %out : f16
                linalg.yield %112 : f16
              } -> tensor<1x32x1xf16>
              %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %97 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%51 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %112 = arith.mulf %in_9, %cst_2 : f16
                %113 = arith.cmpf ogt, %in, %112 : f16
                %114 = arith.select %113, %in, %112 : f16
                linalg.yield %114 : f16
              } -> tensor<1x32x1xf16>
              %99 = loom.broadcast ins(%98 : tensor<1x32x1xf16>) outs(%67 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x64xf16>
              %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %99 : tensor<1x32x64xf16>, tensor<1x32x64xf16>) outs(%62 : tensor<1x32x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %112 = arith.mulf %in, %cst_2 : f16
                %113 = arith.subf %112, %in_9 : f16
                %114 = math.exp %113 : f16
                linalg.yield %114 : f16
              } -> tensor<1x32x64xf16>
              loom.semaphore_give %66 : memref<1x32x32xf16>
              %101 = linalg.fill ins(%cst : f16) outs(%54 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%100 : tensor<1x32x64xf16>) outs(%101 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %112 = arith.addf %in, %out : f16
                linalg.yield %112 : f16
              } -> tensor<1x32x1xf16>
              %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %98 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%57 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %112 = arith.subf %in, %in_9 : f16
                %113 = math.exp %112 : f16
                linalg.yield %113 : f16
              } -> tensor<1x32x1xf16>
              %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %103, %102 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%arg9 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %112 = arith.mulf %in, %in_9 : f16
                %113 = arith.addf %112, %in_10 : f16
                linalg.yield %113 : f16
              } -> tensor<1x32x1xf16>
              loom.semaphore_give %53 : memref<1x32x1xf16>
              %105 = loom.broadcast ins(%103 : tensor<1x32x1xf16>) outs(%65 : tensor<1x32x32xf16>) dim(2) -> tensor<1x32x128xf16>
              loom.semaphore_give %56 : memref<1x32x1xf16>
              %c0_7 = arith.constant 0 : index
              %106 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 1048576 + d1 * 128 + d2)>(%20, %91, %c0_7)
              %reinterpret_cast_8 = memref.reinterpret_cast %arg1 to offset: [%106], sizes: [1, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x64x128xf16>
              %107 = loom.bufferize_to_tensor %69[1, 64, 128] : memref<1x64x128xf16> -> tensor<1x64x128xf16>
              %108 = linalg.fill ins(%cst : f16) outs(%48 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %109 = linalg.batch_matmul ins(%100, %107 : tensor<1x32x64xf16>, tensor<1x64x128xf16>) outs(%108 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              loom.semaphore_give %69 : memref<1x64x128xf16>
              loom.semaphore_give %61 : memref<1x32x64xf16>
              %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%109, %arg10, %105 : tensor<1x32x128xf16>, tensor<1x32x128xf16>, tensor<1x32x128xf16>) outs(%arg10 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %out: f16):
                %112 = arith.mulf %in_9, %in_10 : f16
                %113 = arith.addf %in, %112 : f16
                linalg.yield %113 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %64 : memref<1x32x32xf16>
              loom.semaphore_give %47 : memref<1x32x128xf16>
              %111 = linalg.copy ins(%98 : tensor<1x32x1xf16>) outs(%arg8 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              loom.semaphore_give %50 : memref<1x32x1xf16>
              scf.yield %111, %104, %110 : tensor<1x32x1xf16>, tensor<1x32x1xf16>, tensor<1x32x128xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %24 : memref<1x32x128xf16>
            %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70#1, %70#0 : tensor<1x32x1xf16>, tensor<1x32x1xf16>) outs(%38 : tensor<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %90 = math.log %in : f16
              %91 = arith.addf %90, %in_5 : f16
              linalg.yield %91 : f16
            } -> tensor<1x32x1xf16>
            loom.semaphore_give %39 : memref<1x32x1xf16>
            loom.semaphore_give %43 : memref<1x32x1xf16>
            %72 = loom.alloc [8, 1, 32, 1] on @L1 : memref<8x1x32x1xf16>
            %73 = loom.semaphore_take %72 : memref<8x1x32x1xf16> -> memref<8x1x32x1xf16>
            %74 = loom.init_tensor %73[8, 1, 32, 1] : memref<8x1x32x1xf16> -> tensor<8x1x32x1xf16>
            %75 = loom.semaphore_take %32 : memref<1x32x1xf16> -> memref<1x32x1xf16>
            %76 = loom.init_tensor %75[1, 32, 1] : memref<1x32x1xf16> -> tensor<1x32x1xf16>
            %77 = loom.sync ins(%71 : tensor<1x32x1xf16>) outs(%76 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
            %78 = loom.gather ins(%77 : tensor<1x32x1xf16>) outs(%74 : tensor<8x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x1x32x1xf16>
            loom.semaphore_give %75 : memref<1x32x1xf16>
            loom.semaphore_give %37 : memref<1x32x1xf16>
            %79 = loom.alloc [8, 1, 32, 128] on @L1 : memref<8x1x32x128xf16>
            %80 = loom.semaphore_take %79 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
            %81 = loom.init_tensor %80[8, 1, 32, 128] : memref<8x1x32x128xf16> -> tensor<8x1x32x128xf16>
            %82 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            %83 = loom.init_tensor %82[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
            %84 = loom.sync ins(%70#2 : tensor<1x32x128xf16>) outs(%83 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
            %85 = loom.gather ins(%84 : tensor<1x32x128xf16>) outs(%81 : tensor<8x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) -> tensor<8x1x32x128xf16>
            loom.semaphore_give %82 : memref<1x32x128xf16>
            loom.semaphore_give %29 : memref<1x32x128xf16>
            %86 = loom.alloc [1024, 1, 32, 32] on @L1 : memref<1024x1x32x32xf16>
            %87 = loom.semaphore_take %86 : memref<1024x1x32x32xf16> -> memref<1024x1x32x32xf16>
            %88 = loom.init_tensor %87[1024, 1, 32, 32] : memref<1024x1x32x32xf16> -> tensor<1024x1x32x32xf16>
            %89 = arith.cmpi eq, %arg5, %c0 : index
            scf.if %89 {
              %90 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %91 = loom.semaphore_take %72 : memref<8x1x32x1xf16> -> memref<8x1x32x1xf16>
              %92 = loom.init_tensor %91[8, 1, 32, 1] : memref<8x1x32x1xf16> -> tensor<8x1x32x1xf16>
              %93 = loom.sync ins(%78 : tensor<8x1x32x1xf16>) outs(%92 : tensor<8x1x32x1xf16>) -> tensor<8x1x32x1xf16>
              %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%93 : tensor<8x1x32x1xf16>) outs(%90 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %111 = arith.maximumf %in, %out : f16
                linalg.yield %111 : f16
              } -> tensor<1x32x1xf16>
              %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %94 : tensor<8x1x32x1xf16>, tensor<1x32x1xf16>) outs(%74 : tensor<8x1x32x1xf16>) {
              ^bb0(%in: f16, %in_8: f16, %out: f16):
                %111 = arith.subf %in, %in_8 : f16
                %112 = math.exp %111 : f16
                linalg.yield %112 : f16
              } -> tensor<8x1x32x1xf16>
              loom.semaphore_give %35 : memref<1x32x1xf16>
              %96 = linalg.fill ins(%cst : f16) outs(%34 : tensor<1x32x1xf16>) -> tensor<1x32x1xf16>
              %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%95 : tensor<8x1x32x1xf16>) outs(%96 : tensor<1x32x1xf16>) {
              ^bb0(%in: f16, %out: f16):
                %111 = arith.addf %in, %out : f16
                linalg.yield %111 : f16
              } -> tensor<1x32x1xf16>
              %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%95, %97 : tensor<8x1x32x1xf16>, tensor<1x32x1xf16>) outs(%74 : tensor<8x1x32x1xf16>) {
              ^bb0(%in: f16, %in_8: f16, %out: f16):
                %111 = arith.divf %in, %in_8 : f16
                linalg.yield %111 : f16
              } -> tensor<8x1x32x1xf16>
              loom.semaphore_give %33 : memref<1x32x1xf16>
              %99 = loom.broadcast ins(%98 : tensor<8x1x32x1xf16>) outs(%88 : tensor<1024x1x32x32xf16>) dim(3) -> tensor<8x1x32x128xf16>
              loom.semaphore_give %91 : memref<8x1x32x1xf16>
              loom.semaphore_give %73 : memref<8x1x32x1xf16>
              %100 = loom.semaphore_take %79 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
              %101 = loom.init_tensor %100[8, 1, 32, 128] : memref<8x1x32x128xf16> -> tensor<8x1x32x128xf16>
              %102 = loom.sync ins(%85 : tensor<8x1x32x128xf16>) outs(%101 : tensor<8x1x32x128xf16>) -> tensor<8x1x32x128xf16>
              %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%102, %99 : tensor<8x1x32x128xf16>, tensor<8x1x32x128xf16>) outs(%81 : tensor<8x1x32x128xf16>) {
              ^bb0(%in: f16, %in_8: f16, %out: f16):
                %111 = arith.mulf %in, %in_8 : f16
                linalg.yield %111 : f16
              } -> tensor<8x1x32x128xf16>
              loom.semaphore_give %87 : memref<1024x1x32x32xf16>
              %104 = linalg.fill ins(%cst : f16) outs(%23 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%103 : tensor<8x1x32x128xf16>) outs(%104 : tensor<1x32x128xf16>) {
              ^bb0(%in: f16, %out: f16):
                %111 = arith.addf %in, %out : f16
                linalg.yield %111 : f16
              } -> tensor<1x32x128xf16>
              loom.semaphore_give %100 : memref<8x1x32x128xf16>
              loom.semaphore_give %80 : memref<8x1x32x128xf16>
              %c0_5 = arith.constant 0 : index
              %c0_6 = arith.constant 0 : index
              %106 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4096 + d1 * 128 + d2)>(%20, %c0_5, %c0_6)
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%106], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              %107 = loom.semaphore_take %21 : memref<1x32x128xf16> -> memref<1x32x128xf16>
              %108 = loom.init_tensor %107[1, 32, 128] : memref<1x32x128xf16> -> tensor<1x32x128xf16>
              %109 = loom.sync ins(%105 : tensor<1x32x128xf16>) outs(%108 : tensor<1x32x128xf16>) -> tensor<1x32x128xf16>
              %110 = loom.bufferize_to_memref %109 : tensor<1x32x128xf16> -> memref<1x32x128xf16>
              loom.copy %110, %reinterpret_cast_7 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
              loom.semaphore_give %107 : memref<1x32x128xf16>
              loom.semaphore_give %22 : memref<1x32x128xf16>
            }
          } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
