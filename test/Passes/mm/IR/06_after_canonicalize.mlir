module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y1y8__d0i0_d1i0_d2i1__f01__dim_y_level1_bc8_dim_x_level0_bc8_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (1) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c512 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %19 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %20 = loom.semaphore_take %19 : memref<1x512xf16> -> memref<1x512xf16>
              %21 = loom.init_tensor %20[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %23 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %24 = loom.semaphore_take %23 : memref<1x512xf16> -> memref<1x512xf16>
              %25 = loom.init_tensor %24[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %26 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %27 = loom.semaphore_take %26 : memref<1x512xf16> -> memref<1x512xf16>
              %28 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %29 = loom.semaphore_take %28 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg3, %c0], LR : [%arg3, %c7]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %31 = loom.bufferize_to_tensor %27[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %32 = arith.muli %arg5, %c512 : index
              %c0_1 = arith.constant 0 : index
              %33 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %32)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              %34 = arith.addi %arg4, %arg5 : index
              loom.copy %reinterpret_cast_2, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %34], LR : [%c7, %34]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %35 = loom.bufferize_to_tensor %29[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %36 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %37 = linalg.matmul ins(%31, %35 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%36 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %29 : memref<512x512xf16>
              loom.semaphore_give %27 : memref<1x512xf16>
              %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %37 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%22 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %47 = arith.addf %in, %in_4 : f16
                linalg.yield %47 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %24 : memref<1x512xf16>
              %39 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %40 = loom.semaphore_take %39 : memref<1x512xf16> -> memref<1x512xf16>
              %41 = loom.init_tensor %40[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %42 = linalg.copy ins(%38 : tensor<1x512xf16>) outs(%41 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %20 : memref<1x512xf16>
              %43 = arith.muli %arg5, %c512 : index
              %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %43)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %45 = loom.bufferize_to_memref %42 : tensor<1x512xf16> -> memref<1x512xf16>
              %46 = arith.addi %arg4, %arg5 : index
              loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %46], LR : [%arg3, %46]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %40 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y1y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            scf.for %arg6 = %c0 to %c512 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %19 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %20 = loom.semaphore_take %19 : memref<1x512xf16> -> memref<1x512xf16>
              %21 = loom.init_tensor %20[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %23 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %24 = loom.semaphore_take %23 : memref<1x512xf16> -> memref<1x512xf16>
              %25 = loom.init_tensor %24[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %26 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %27 = loom.semaphore_take %26 : memref<1x512xf16> -> memref<1x512xf16>
              %28 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %29 = loom.semaphore_take %28 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              %31 = arith.addi %arg5, %arg3 : index
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %31], LR : [%c7, %31]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %32 = loom.bufferize_to_tensor %27[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %33 = arith.muli %arg4, %c512 : index
              %c0_1 = arith.constant 0 : index
              %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_2, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %35 = loom.bufferize_to_tensor %29[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %36 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %37 = linalg.matmul ins(%32, %35 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%36 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %29 : memref<512x512xf16>
              loom.semaphore_give %27 : memref<1x512xf16>
              %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %37 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%22 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %47 = arith.addf %in, %in_4 : f16
                linalg.yield %47 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %24 : memref<1x512xf16>
              %39 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %40 = loom.semaphore_take %39 : memref<1x512xf16> -> memref<1x512xf16>
              %41 = loom.init_tensor %40[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %42 = linalg.copy ins(%38 : tensor<1x512xf16>) outs(%41 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %20 : memref<1x512xf16>
              %43 = arith.muli %arg4, %c512 : index
              %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %43)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %45 = loom.bufferize_to_memref %42 : tensor<1x512xf16> -> memref<1x512xf16>
              %46 = arith.addi %arg5, %arg3 : index
              loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %46], LR : [%arg4, %46]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %40 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y2y4__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc8_dim_y_level0_bc2_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c256 = arith.constant 256 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c256 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg3, %arg4, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %20 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %21 = loom.semaphore_take %20 : memref<1x512xf16> -> memref<1x512xf16>
                %22 = loom.init_tensor %21[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %24 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %25 = loom.semaphore_take %24 : memref<1x512xf16> -> memref<1x512xf16>
                %26 = loom.init_tensor %25[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %27 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %28 = loom.semaphore_take %27 : memref<1x512xf16> -> memref<1x512xf16>
                %29 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
                %30 = loom.semaphore_take %29 : memref<512x512xf16> -> memref<512x512xf16>
                %c0_0 = arith.constant 0 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
                %32 = arith.muli %arg5, %c2 : index
                %33 = arith.addi %arg4, %32 : index
                loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %33], LR : [%arg3, %33]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
                %34 = loom.bufferize_to_tensor %28[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %35 = arith.muli %19, %c512 : index
                %c0_1 = arith.constant 0 : index
                %36 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %35)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
                %37 = arith.addi %32, %c1 : index
                loom.copy %reinterpret_cast_2, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %32], LR : [%c7, %37]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
                %38 = loom.bufferize_to_tensor %30[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %40 = linalg.matmul ins(%34, %38 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%39 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %30 : memref<512x512xf16>
                loom.semaphore_give %28 : memref<1x512xf16>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %40 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%23 : tensor<1x512xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %51 = arith.addf %in, %in_4 : f16
                  linalg.yield %51 : f16
                } -> tensor<1x512xf16>
                loom.semaphore_give %25 : memref<1x512xf16>
                %42 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %43 = loom.semaphore_take %42 : memref<1x512xf16> -> memref<1x512xf16>
                %44 = loom.init_tensor %43[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %45 = linalg.copy ins(%41 : tensor<1x512xf16>) outs(%44 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %21 : memref<1x512xf16>
                %46 = arith.muli %19, %c512 : index
                %47 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %46)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%47], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                %48 = loom.bufferize_to_memref %45 : tensor<1x512xf16> -> memref<1x512xf16>
                %49 = arith.muli %arg5, %c2 : index
                %50 = arith.addi %arg4, %49 : index
                loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %50], LR : [%arg3, %50]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %43 : memref<1x512xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y2y4__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc2_n_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c1024 = arith.constant 1024 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c1024 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg4, %arg5)
              %20 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %21 = loom.semaphore_take %20 : memref<1x512xf16> -> memref<1x512xf16>
              %22 = loom.init_tensor %21[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %24 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %25 = loom.semaphore_take %24 : memref<1x512xf16> -> memref<1x512xf16>
              %26 = loom.init_tensor %25[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %27 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %28 = loom.semaphore_take %27 : memref<1x512xf16> -> memref<1x512xf16>
              %29 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %30 = loom.semaphore_take %29 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              %32 = arith.muli %arg3, %c2 : index
              %33 = arith.addi %32, %c1 : index
              loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %32], LR : [%c7, %33]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %34 = loom.bufferize_to_tensor %28[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %35 = arith.muli %19, %c512 : index
              %c0_1 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %35)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              %37 = arith.addi %arg5, %32 : index
              loom.copy %reinterpret_cast_2, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %37], LR : [%arg4, %37]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %38 = loom.bufferize_to_tensor %30[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %39 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %40 = linalg.matmul ins(%34, %38 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%39 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %30 : memref<512x512xf16>
              loom.semaphore_give %28 : memref<1x512xf16>
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %40 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%23 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %51 = arith.addf %in, %in_4 : f16
                linalg.yield %51 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %25 : memref<1x512xf16>
              %42 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %43 = loom.semaphore_take %42 : memref<1x512xf16> -> memref<1x512xf16>
              %44 = loom.init_tensor %43[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %45 = linalg.copy ins(%41 : tensor<1x512xf16>) outs(%44 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %21 : memref<1x512xf16>
              %46 = arith.muli %19, %c512 : index
              %47 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %46)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%47], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %48 = loom.bufferize_to_memref %45 : tensor<1x512xf16> -> memref<1x512xf16>
              %49 = arith.muli %arg3, %c2 : index
              %50 = arith.addi %arg5, %49 : index
              loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %50], LR : [%arg4, %50]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %43 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y4y2__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc8_dim_y_level0_bc4_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c128 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg3, %arg4, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %20 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %21 = loom.semaphore_take %20 : memref<1x512xf16> -> memref<1x512xf16>
                %22 = loom.init_tensor %21[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %24 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %25 = loom.semaphore_take %24 : memref<1x512xf16> -> memref<1x512xf16>
                %26 = loom.init_tensor %25[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %27 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %28 = loom.semaphore_take %27 : memref<1x512xf16> -> memref<1x512xf16>
                %29 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
                %30 = loom.semaphore_take %29 : memref<512x512xf16> -> memref<512x512xf16>
                %c0_0 = arith.constant 0 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
                %32 = arith.muli %arg5, %c4 : index
                %33 = arith.addi %arg4, %32 : index
                loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %33], LR : [%arg3, %33]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
                %34 = loom.bufferize_to_tensor %28[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %35 = arith.muli %19, %c512 : index
                %c0_1 = arith.constant 0 : index
                %36 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %35)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
                %37 = arith.addi %32, %c3 : index
                loom.copy %reinterpret_cast_2, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %37]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
                %38 = loom.bufferize_to_tensor %30[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %40 = linalg.matmul ins(%34, %38 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%39 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %30 : memref<512x512xf16>
                loom.semaphore_give %28 : memref<1x512xf16>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %40 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%23 : tensor<1x512xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %51 = arith.addf %in, %in_4 : f16
                  linalg.yield %51 : f16
                } -> tensor<1x512xf16>
                loom.semaphore_give %25 : memref<1x512xf16>
                %42 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %43 = loom.semaphore_take %42 : memref<1x512xf16> -> memref<1x512xf16>
                %44 = loom.init_tensor %43[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %45 = linalg.copy ins(%41 : tensor<1x512xf16>) outs(%44 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %21 : memref<1x512xf16>
                %46 = arith.muli %19, %c512 : index
                %47 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %46)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%47], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                %48 = loom.bufferize_to_memref %45 : tensor<1x512xf16> -> memref<1x512xf16>
                %49 = arith.muli %arg5, %c4 : index
                %50 = arith.addi %arg4, %49 : index
                loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %50], LR : [%arg3, %50]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %43 : memref<1x512xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y4y2__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc4_n_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2048 = arith.constant 2048 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c2048 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg4, %arg5)
              %20 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %21 = loom.semaphore_take %20 : memref<1x512xf16> -> memref<1x512xf16>
              %22 = loom.init_tensor %21[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %24 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %25 = loom.semaphore_take %24 : memref<1x512xf16> -> memref<1x512xf16>
              %26 = loom.init_tensor %25[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %27 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %28 = loom.semaphore_take %27 : memref<1x512xf16> -> memref<1x512xf16>
              %29 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %30 = loom.semaphore_take %29 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              %32 = arith.muli %arg3, %c4 : index
              %33 = arith.addi %32, %c3 : index
              loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %33]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %34 = loom.bufferize_to_tensor %28[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %35 = arith.muli %19, %c512 : index
              %c0_1 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %35)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              %37 = arith.addi %arg5, %32 : index
              loom.copy %reinterpret_cast_2, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %37], LR : [%arg4, %37]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %38 = loom.bufferize_to_tensor %30[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %39 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %40 = linalg.matmul ins(%34, %38 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%39 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %30 : memref<512x512xf16>
              loom.semaphore_give %28 : memref<1x512xf16>
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %40 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%23 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %51 = arith.addf %in, %in_4 : f16
                linalg.yield %51 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %25 : memref<1x512xf16>
              %42 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %43 = loom.semaphore_take %42 : memref<1x512xf16> -> memref<1x512xf16>
              %44 = loom.init_tensor %43[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %45 = linalg.copy ins(%41 : tensor<1x512xf16>) outs(%44 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %21 : memref<1x512xf16>
              %46 = arith.muli %19, %c512 : index
              %47 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %46)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%47], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %48 = loom.bufferize_to_memref %45 : tensor<1x512xf16> -> memref<1x512xf16>
              %49 = arith.muli %arg3, %c4 : index
              %50 = arith.addi %arg5, %49 : index
              loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %50], LR : [%arg4, %50]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %43 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y8y1__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc8_dim_y_level1_bc8_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c64 = arith.constant 64 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              scf.for %arg7 = %c0 to %c8 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
                %19 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %20 = loom.semaphore_take %19 : memref<1x512xf16> -> memref<1x512xf16>
                %21 = loom.init_tensor %20[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %23 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %24 = loom.semaphore_take %23 : memref<1x512xf16> -> memref<1x512xf16>
                %25 = loom.init_tensor %24[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %26 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %27 = loom.semaphore_take %26 : memref<1x512xf16> -> memref<1x512xf16>
                %28 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
                %29 = loom.semaphore_take %28 : memref<512x512xf16> -> memref<512x512xf16>
                %c0_0 = arith.constant 0 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
                %31 = arith.muli %arg5, %c8 : index
                %32 = arith.addi %arg4, %31 : index
                loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %32], LR : [%arg3, %32]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
                %33 = loom.bufferize_to_tensor %27[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %34 = arith.muli %arg7, %c512 : index
                %c0_1 = arith.constant 0 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %34)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%35], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_2, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
                %36 = loom.bufferize_to_tensor %29[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %38 = linalg.matmul ins(%33, %36 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%37 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %29 : memref<512x512xf16>
                loom.semaphore_give %27 : memref<1x512xf16>
                %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %38 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%22 : tensor<1x512xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %49 = arith.addf %in, %in_4 : f16
                  linalg.yield %49 : f16
                } -> tensor<1x512xf16>
                loom.semaphore_give %24 : memref<1x512xf16>
                %40 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %41 = loom.semaphore_take %40 : memref<1x512xf16> -> memref<1x512xf16>
                %42 = loom.init_tensor %41[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %43 = linalg.copy ins(%39 : tensor<1x512xf16>) outs(%42 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %20 : memref<1x512xf16>
                %44 = arith.muli %arg7, %c512 : index
                %45 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %44)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%45], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                %46 = loom.bufferize_to_memref %43 : tensor<1x512xf16> -> memref<1x512xf16>
                %47 = arith.muli %arg5, %c8 : index
                %48 = arith.addi %arg4, %47 : index
                loom.copy %46, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %48], LR : [%arg3, %48]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<1x512xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y8y1__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4096 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg4, %arg5)
              %19 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %20 = loom.semaphore_take %19 : memref<1x512xf16> -> memref<1x512xf16>
              %21 = loom.init_tensor %20[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %23 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %24 = loom.semaphore_take %23 : memref<1x512xf16> -> memref<1x512xf16>
              %25 = loom.init_tensor %24[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %26 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %27 = loom.semaphore_take %26 : memref<1x512xf16> -> memref<1x512xf16>
              %28 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %29 = loom.semaphore_take %28 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %31 = loom.bufferize_to_tensor %27[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %32 = arith.muli %18, %c512 : index
              %c0_1 = arith.constant 0 : index
              %33 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %32)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              %34 = arith.muli %arg3, %c8 : index
              %35 = arith.addi %arg5, %34 : index
              loom.copy %reinterpret_cast_2, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %35], LR : [%arg4, %35]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %36 = loom.bufferize_to_tensor %29[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %37 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %38 = linalg.matmul ins(%31, %36 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%37 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %29 : memref<512x512xf16>
              loom.semaphore_give %27 : memref<1x512xf16>
              %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %38 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%22 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %49 = arith.addf %in, %in_4 : f16
                linalg.yield %49 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %24 : memref<1x512xf16>
              %40 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %41 = loom.semaphore_take %40 : memref<1x512xf16> -> memref<1x512xf16>
              %42 = loom.init_tensor %41[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %43 = linalg.copy ins(%39 : tensor<1x512xf16>) outs(%42 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %20 : memref<1x512xf16>
              %44 = arith.muli %18, %c512 : index
              %45 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %44)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%45], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %46 = loom.bufferize_to_memref %43 : tensor<1x512xf16> -> memref<1x512xf16>
              %47 = arith.muli %arg3, %c8 : index
              %48 = arith.addi %arg5, %47 : index
              loom.copy %46, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %48], LR : [%arg4, %48]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %41 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x1x8_y8__d0i0_d1i0_d2i1__f01__dim_x_level1_bc8_dim_y_level0_bc8_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c512 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %19 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %20 = loom.semaphore_take %19 : memref<1x512xf16> -> memref<1x512xf16>
              %21 = loom.init_tensor %20[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %23 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %24 = loom.semaphore_take %23 : memref<1x512xf16> -> memref<1x512xf16>
              %25 = loom.init_tensor %24[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %26 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %27 = loom.semaphore_take %26 : memref<1x512xf16> -> memref<1x512xf16>
              %28 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %29 = loom.semaphore_take %28 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %31 = loom.bufferize_to_tensor %27[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %32 = arith.muli %arg5, %c512 : index
              %c0_1 = arith.constant 0 : index
              %33 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %32)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              %34 = arith.addi %arg3, %arg5 : index
              loom.copy %reinterpret_cast_2, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%34, %c0], LR : [%34, %c7]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %35 = loom.bufferize_to_tensor %29[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %36 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %37 = linalg.matmul ins(%31, %35 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%36 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %29 : memref<512x512xf16>
              loom.semaphore_give %27 : memref<1x512xf16>
              %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %37 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%22 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %47 = arith.addf %in, %in_4 : f16
                linalg.yield %47 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %24 : memref<1x512xf16>
              %39 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %40 = loom.semaphore_take %39 : memref<1x512xf16> -> memref<1x512xf16>
              %41 = loom.init_tensor %40[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %42 = linalg.copy ins(%38 : tensor<1x512xf16>) outs(%41 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %20 : memref<1x512xf16>
              %43 = arith.muli %arg5, %c512 : index
              %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %43)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %45 = loom.bufferize_to_memref %42 : tensor<1x512xf16> -> memref<1x512xf16>
              %46 = arith.addi %arg3, %arg5 : index
              loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%46, %arg4], LR : [%46, %arg4]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %40 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x1x8_y8__d0i1_d1i1_d2i0__f01__dim_y_level0_bc8_dim_x_level1_bc8_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (1) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c512 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %19 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %20 = loom.semaphore_take %19 : memref<1x512xf16> -> memref<1x512xf16>
              %21 = loom.init_tensor %20[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %23 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %24 = loom.semaphore_take %23 : memref<1x512xf16> -> memref<1x512xf16>
              %25 = loom.init_tensor %24[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %26 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %27 = loom.semaphore_take %26 : memref<1x512xf16> -> memref<1x512xf16>
              %28 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %29 = loom.semaphore_take %28 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              %31 = arith.addi %arg4, %arg3 : index
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%31, %c0], LR : [%31, %c7]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %32 = loom.bufferize_to_tensor %27[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %33 = arith.muli %arg5, %c512 : index
              %c0_1 = arith.constant 0 : index
              %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_2, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg5], LR : [%c7, %arg5]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %35 = loom.bufferize_to_tensor %29[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %36 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %37 = linalg.matmul ins(%32, %35 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%36 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %29 : memref<512x512xf16>
              loom.semaphore_give %27 : memref<1x512xf16>
              %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %37 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%22 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %47 = arith.addf %in, %in_4 : f16
                linalg.yield %47 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %24 : memref<1x512xf16>
              %39 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %40 = loom.semaphore_take %39 : memref<1x512xf16> -> memref<1x512xf16>
              %41 = loom.init_tensor %40[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %42 = linalg.copy ins(%38 : tensor<1x512xf16>) outs(%41 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %20 : memref<1x512xf16>
              %43 = arith.muli %arg5, %c512 : index
              %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %43)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %45 = loom.bufferize_to_memref %42 : tensor<1x512xf16> -> memref<1x512xf16>
              %46 = arith.addi %arg4, %arg3 : index
              loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%46, %arg5], LR : [%46, %arg5]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %40 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x2x4_y8__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc2_dim_y_level0_bc8_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c256 = arith.constant 256 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c256 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg3, %arg4, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %20 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %21 = loom.semaphore_take %20 : memref<1x512xf16> -> memref<1x512xf16>
                %22 = loom.init_tensor %21[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %24 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %25 = loom.semaphore_take %24 : memref<1x512xf16> -> memref<1x512xf16>
                %26 = loom.init_tensor %25[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %27 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %28 = loom.semaphore_take %27 : memref<1x512xf16> -> memref<1x512xf16>
                %29 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
                %30 = loom.semaphore_take %29 : memref<512x512xf16> -> memref<512x512xf16>
                %c0_0 = arith.constant 0 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
                %32 = arith.muli %arg5, %c2 : index
                %33 = arith.addi %arg3, %32 : index
                loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%33, %arg4], LR : [%33, %arg4]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
                %34 = loom.bufferize_to_tensor %28[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %35 = arith.muli %19, %c512 : index
                %c0_1 = arith.constant 0 : index
                %36 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %35)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
                %37 = arith.addi %32, %c1 : index
                loom.copy %reinterpret_cast_2, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%32, %c0], LR : [%37, %c7]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
                %38 = loom.bufferize_to_tensor %30[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %40 = linalg.matmul ins(%34, %38 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%39 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %30 : memref<512x512xf16>
                loom.semaphore_give %28 : memref<1x512xf16>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %40 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%23 : tensor<1x512xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %51 = arith.addf %in, %in_4 : f16
                  linalg.yield %51 : f16
                } -> tensor<1x512xf16>
                loom.semaphore_give %25 : memref<1x512xf16>
                %42 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %43 = loom.semaphore_take %42 : memref<1x512xf16> -> memref<1x512xf16>
                %44 = loom.init_tensor %43[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %45 = linalg.copy ins(%41 : tensor<1x512xf16>) outs(%44 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %21 : memref<1x512xf16>
                %46 = arith.muli %19, %c512 : index
                %47 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %46)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%47], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                %48 = loom.bufferize_to_memref %45 : tensor<1x512xf16> -> memref<1x512xf16>
                %49 = arith.muli %arg5, %c2 : index
                %50 = arith.addi %arg3, %49 : index
                loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%50, %arg4], LR : [%50, %arg4]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %43 : memref<1x512xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x2x4_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc2_dim_y_level0_bc8_n_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c1024 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%arg4, %arg5)
              %20 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %21 = loom.semaphore_take %20 : memref<1x512xf16> -> memref<1x512xf16>
              %22 = loom.init_tensor %21[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %24 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %25 = loom.semaphore_take %24 : memref<1x512xf16> -> memref<1x512xf16>
              %26 = loom.init_tensor %25[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %27 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %28 = loom.semaphore_take %27 : memref<1x512xf16> -> memref<1x512xf16>
              %29 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %30 = loom.semaphore_take %29 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              %32 = arith.muli %arg3, %c2 : index
              %33 = arith.addi %32, %c1 : index
              loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%32, %c0], LR : [%33, %c7]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %34 = loom.bufferize_to_tensor %28[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %35 = arith.muli %19, %c512 : index
              %c0_1 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %35)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              %37 = arith.addi %arg4, %32 : index
              loom.copy %reinterpret_cast_2, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%37, %arg5], LR : [%37, %arg5]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %38 = loom.bufferize_to_tensor %30[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %39 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %40 = linalg.matmul ins(%34, %38 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%39 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %30 : memref<512x512xf16>
              loom.semaphore_give %28 : memref<1x512xf16>
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %40 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%23 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %51 = arith.addf %in, %in_4 : f16
                linalg.yield %51 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %25 : memref<1x512xf16>
              %42 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %43 = loom.semaphore_take %42 : memref<1x512xf16> -> memref<1x512xf16>
              %44 = loom.init_tensor %43[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %45 = linalg.copy ins(%41 : tensor<1x512xf16>) outs(%44 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %21 : memref<1x512xf16>
              %46 = arith.muli %19, %c512 : index
              %47 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %46)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%47], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %48 = loom.bufferize_to_memref %45 : tensor<1x512xf16> -> memref<1x512xf16>
              %49 = arith.muli %arg3, %c2 : index
              %50 = arith.addi %arg4, %49 : index
              loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%50, %arg5], LR : [%50, %arg5]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %43 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x4x2_y8__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc4_dim_y_level0_bc8_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c128 = arith.constant 128 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c128 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg3, %arg4, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %20 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %21 = loom.semaphore_take %20 : memref<1x512xf16> -> memref<1x512xf16>
                %22 = loom.init_tensor %21[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %24 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %25 = loom.semaphore_take %24 : memref<1x512xf16> -> memref<1x512xf16>
                %26 = loom.init_tensor %25[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %27 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %28 = loom.semaphore_take %27 : memref<1x512xf16> -> memref<1x512xf16>
                %29 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
                %30 = loom.semaphore_take %29 : memref<512x512xf16> -> memref<512x512xf16>
                %c0_0 = arith.constant 0 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
                %32 = arith.muli %arg5, %c4 : index
                %33 = arith.addi %arg3, %32 : index
                loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%33, %arg4], LR : [%33, %arg4]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
                %34 = loom.bufferize_to_tensor %28[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %35 = arith.muli %19, %c512 : index
                %c0_1 = arith.constant 0 : index
                %36 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %35)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
                %37 = arith.addi %32, %c3 : index
                loom.copy %reinterpret_cast_2, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%32, %c0], LR : [%37, %c7]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
                %38 = loom.bufferize_to_tensor %30[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
                %39 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %40 = linalg.matmul ins(%34, %38 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%39 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %30 : memref<512x512xf16>
                loom.semaphore_give %28 : memref<1x512xf16>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %40 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%23 : tensor<1x512xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %51 = arith.addf %in, %in_4 : f16
                  linalg.yield %51 : f16
                } -> tensor<1x512xf16>
                loom.semaphore_give %25 : memref<1x512xf16>
                %42 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %43 = loom.semaphore_take %42 : memref<1x512xf16> -> memref<1x512xf16>
                %44 = loom.init_tensor %43[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %45 = linalg.copy ins(%41 : tensor<1x512xf16>) outs(%44 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %21 : memref<1x512xf16>
                %46 = arith.muli %19, %c512 : index
                %47 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %46)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%47], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                %48 = loom.bufferize_to_memref %45 : tensor<1x512xf16> -> memref<1x512xf16>
                %49 = arith.muli %arg5, %c4 : index
                %50 = arith.addi %arg3, %49 : index
                loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%50, %arg4], LR : [%50, %arg4]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %43 : memref<1x512xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x4x2_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc4_dim_y_level0_bc8_n_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2048 = arith.constant 2048 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c2048 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
              %19 = affine.apply affine_map<(d0, d1) -> (d0 * 4 + d1)>(%arg4, %arg5)
              %20 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %21 = loom.semaphore_take %20 : memref<1x512xf16> -> memref<1x512xf16>
              %22 = loom.init_tensor %21[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %24 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %25 = loom.semaphore_take %24 : memref<1x512xf16> -> memref<1x512xf16>
              %26 = loom.init_tensor %25[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %27 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %28 = loom.semaphore_take %27 : memref<1x512xf16> -> memref<1x512xf16>
              %29 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %30 = loom.semaphore_take %29 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%31], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              %32 = arith.muli %arg3, %c4 : index
              %33 = arith.addi %32, %c3 : index
              loom.copy %reinterpret_cast, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%32, %c0], LR : [%33, %c7]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %34 = loom.bufferize_to_tensor %28[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %35 = arith.muli %19, %c512 : index
              %c0_1 = arith.constant 0 : index
              %36 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %35)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%36], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              %37 = arith.addi %arg4, %32 : index
              loom.copy %reinterpret_cast_2, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%37, %arg5], LR : [%37, %arg5]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %38 = loom.bufferize_to_tensor %30[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %39 = linalg.fill ins(%cst : f16) outs(%26 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %40 = linalg.matmul ins(%34, %38 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%39 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %30 : memref<512x512xf16>
              loom.semaphore_give %28 : memref<1x512xf16>
              %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%23, %40 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%23 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %51 = arith.addf %in, %in_4 : f16
                linalg.yield %51 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %25 : memref<1x512xf16>
              %42 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %43 = loom.semaphore_take %42 : memref<1x512xf16> -> memref<1x512xf16>
              %44 = loom.init_tensor %43[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %45 = linalg.copy ins(%41 : tensor<1x512xf16>) outs(%44 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %21 : memref<1x512xf16>
              %46 = arith.muli %19, %c512 : index
              %47 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %46)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%47], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %48 = loom.bufferize_to_memref %45 : tensor<1x512xf16> -> memref<1x512xf16>
              %49 = arith.muli %arg3, %c4 : index
              %50 = arith.addi %arg4, %49 : index
              loom.copy %48, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%50, %arg5], LR : [%50, %arg5]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %43 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8x1_y8__d0i0_d1i0_d2i1__f01__n_dim_x_level1_bc8_dim_y_level0_bc8_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c64 = arith.constant 64 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              scf.for %arg7 = %c0 to %c8 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
                %19 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %20 = loom.semaphore_take %19 : memref<1x512xf16> -> memref<1x512xf16>
                %21 = loom.init_tensor %20[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %23 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %24 = loom.semaphore_take %23 : memref<1x512xf16> -> memref<1x512xf16>
                %25 = loom.init_tensor %24[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %26 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %27 = loom.semaphore_take %26 : memref<1x512xf16> -> memref<1x512xf16>
                %28 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
                %29 = loom.semaphore_take %28 : memref<512x512xf16> -> memref<512x512xf16>
                %c0_0 = arith.constant 0 : index
                %30 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%18, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
                %31 = arith.muli %arg5, %c8 : index
                %32 = arith.addi %arg3, %31 : index
                loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%32, %arg4], LR : [%32, %arg4]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
                %33 = loom.bufferize_to_tensor %27[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %34 = arith.muli %arg7, %c512 : index
                %c0_1 = arith.constant 0 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %34)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%35], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_2, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
                %36 = loom.bufferize_to_tensor %29[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
                %37 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x512xf16>) -> tensor<1x512xf16>
                %38 = linalg.matmul ins(%33, %36 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%37 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %29 : memref<512x512xf16>
                loom.semaphore_give %27 : memref<1x512xf16>
                %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %38 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%22 : tensor<1x512xf16>) {
                ^bb0(%in: f16, %in_4: f16, %out: f16):
                  %49 = arith.addf %in, %in_4 : f16
                  linalg.yield %49 : f16
                } -> tensor<1x512xf16>
                loom.semaphore_give %24 : memref<1x512xf16>
                %40 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
                %41 = loom.semaphore_take %40 : memref<1x512xf16> -> memref<1x512xf16>
                %42 = loom.init_tensor %41[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
                %43 = linalg.copy ins(%39 : tensor<1x512xf16>) outs(%42 : tensor<1x512xf16>) -> tensor<1x512xf16>
                loom.semaphore_give %20 : memref<1x512xf16>
                %44 = arith.muli %arg7, %c512 : index
                %45 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%18, %44)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%45], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                %46 = loom.bufferize_to_memref %43 : tensor<1x512xf16> -> memref<1x512xf16>
                %47 = arith.muli %arg5, %c8 : index
                %48 = arith.addi %arg3, %47 : index
                loom.copy %46, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%48, %arg4], LR : [%48, %arg4]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<1x512xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8x1_y8__d0i1_d1i1_d2i0__f01__dim_x_level1_bc8_dim_y_level0_bc8_n_n__tile_k512__tile_m1__tile_n512(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c4096 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg4, %arg5)
              %19 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %20 = loom.semaphore_take %19 : memref<1x512xf16> -> memref<1x512xf16>
              %21 = loom.init_tensor %20[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %23 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %24 = loom.semaphore_take %23 : memref<1x512xf16> -> memref<1x512xf16>
              %25 = loom.init_tensor %24[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %26 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %27 = loom.semaphore_take %26 : memref<1x512xf16> -> memref<1x512xf16>
              %28 = loom.alloc [512, 512] on @L1 : memref<512x512xf16>
              %29 = loom.semaphore_take %28 : memref<512x512xf16> -> memref<512x512xf16>
              %c0_0 = arith.constant 0 : index
              %30 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [1, 512], strides: [512, 1] : memref<4096x512xf16> to memref<1x512xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x512xf16, strided<[512, 1], offset: ?>> to memref<1x512xf16>
              %31 = loom.bufferize_to_tensor %27[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %32 = arith.muli %18, %c512 : index
              %c0_1 = arith.constant 0 : index
              %33 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %32)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 512], strides: [4096, 1] : memref<512x4096xf16> to memref<512x512xf16, strided<[4096, 1], offset: ?>>
              %34 = arith.muli %arg3, %c8 : index
              %35 = arith.addi %arg4, %34 : index
              loom.copy %reinterpret_cast_2, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%35, %arg5], LR : [%35, %arg5]) : memref<512x512xf16, strided<[4096, 1], offset: ?>> to memref<512x512xf16>
              %36 = loom.bufferize_to_tensor %29[512, 512] : memref<512x512xf16> -> tensor<512x512xf16>
              %37 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x512xf16>) -> tensor<1x512xf16>
              %38 = linalg.matmul ins(%31, %36 : tensor<1x512xf16>, tensor<512x512xf16>) outs(%37 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %29 : memref<512x512xf16>
              loom.semaphore_give %27 : memref<1x512xf16>
              %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%22, %38 : tensor<1x512xf16>, tensor<1x512xf16>) outs(%22 : tensor<1x512xf16>) {
              ^bb0(%in: f16, %in_4: f16, %out: f16):
                %49 = arith.addf %in, %in_4 : f16
                linalg.yield %49 : f16
              } -> tensor<1x512xf16>
              loom.semaphore_give %24 : memref<1x512xf16>
              %40 = loom.alloc [1, 512] on @L1 : memref<1x512xf16>
              %41 = loom.semaphore_take %40 : memref<1x512xf16> -> memref<1x512xf16>
              %42 = loom.init_tensor %41[1, 512] : memref<1x512xf16> -> tensor<1x512xf16>
              %43 = linalg.copy ins(%39 : tensor<1x512xf16>) outs(%42 : tensor<1x512xf16>) -> tensor<1x512xf16>
              loom.semaphore_give %20 : memref<1x512xf16>
              %44 = arith.muli %18, %c512 : index
              %45 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %44)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%45], sizes: [1, 512], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              %46 = loom.bufferize_to_memref %43 : tensor<1x512xf16> -> memref<1x512xf16>
              %47 = arith.muli %arg3, %c8 : index
              %48 = arith.addi %arg4, %47 : index
              loom.copy %46, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%48, %arg5], LR : [%48, %arg5]) : memref<1x512xf16> to memref<1x512xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %41 : memref<1x512xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
