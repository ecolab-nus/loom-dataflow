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
    func.func @matmul__x8_y1y8__d0i0_d1i0_d2i1__f01__dim_y_level1_bc8_dim_x_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (1) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              scf.for %arg7 = %c0 to %c8 step %c1 {
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg3, %c0], LR : [%arg3, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %30 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %31 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %31)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %33 = arith.addi %arg4, %arg5 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %33], LR : [%c7, %33]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %34 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %35 = linalg.matmul ins(%30, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %39 = linalg.copy ins(%35 : tensor<64x64xf16>) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %40 = arith.muli %18, %c64 : index
                %41 = arith.muli %19, %c64 : index
                %42 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%40, %41)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %43 = loom.bufferize_to_memref %39 : tensor<64x64xf16> -> memref<64x64xf16>
                %44 = arith.addi %arg4, %arg5 : index
                loom.copy %43, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %44], LR : [%arg3, %44]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %37 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y1y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              scf.for %arg7 = %c0 to %c8 step %c1 {
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.addi %arg5, %arg3 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %30], LR : [%c7, %30]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %31 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %32 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %32)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %34 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %39 = linalg.copy ins(%35 : tensor<64x64xf16>) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %40 = arith.muli %18, %c64 : index
                %41 = arith.muli %19, %c64 : index
                %42 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%40, %41)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %43 = loom.bufferize_to_memref %39 : tensor<64x64xf16> -> memref<64x64xf16>
                %44 = arith.addi %arg5, %arg3 : index
                loom.copy %43, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %44], LR : [%arg4, %44]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %37 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y2y4__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc8_dim_y_level0_bc2_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c16 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg3, %arg4, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.muli %arg5, %c2 : index
                %31 = arith.addi %arg4, %30 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %31], LR : [%arg3, %31]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %32 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %33 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %35 = arith.addi %30, %c1 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %30], LR : [%c7, %35]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %36 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %37 = linalg.matmul ins(%32, %36 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.copy ins(%37 : tensor<64x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %42 = arith.muli %18, %c64 : index
                %43 = arith.muli %19, %c64 : index
                %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%42, %43)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<64x64xf16> -> memref<64x64xf16>
                %46 = arith.muli %arg5, %c2 : index
                %47 = arith.addi %arg4, %46 : index
                loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %47], LR : [%arg3, %47]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y2y4__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc2_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c16 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %19 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg4, %arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.muli %arg3, %c2 : index
                %31 = arith.addi %30, %c1 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %30], LR : [%c7, %31]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %32 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %33 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %35 = arith.addi %arg5, %30 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %35], LR : [%arg4, %35]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %36 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %37 = linalg.matmul ins(%32, %36 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.copy ins(%37 : tensor<64x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %42 = arith.muli %18, %c64 : index
                %43 = arith.muli %19, %c64 : index
                %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%42, %43)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<64x64xf16> -> memref<64x64xf16>
                %46 = arith.muli %arg3, %c2 : index
                %47 = arith.addi %arg5, %46 : index
                loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %47], LR : [%arg4, %47]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y4y2__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc8_dim_y_level0_bc4_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c32 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg3, %arg4, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.muli %arg5, %c4 : index
                %31 = arith.addi %arg4, %30 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %31], LR : [%arg3, %31]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %32 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %33 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %35 = arith.addi %30, %c3 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %30], LR : [%c7, %35]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %36 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %37 = linalg.matmul ins(%32, %36 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.copy ins(%37 : tensor<64x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %42 = arith.muli %18, %c64 : index
                %43 = arith.muli %19, %c64 : index
                %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%42, %43)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<64x64xf16> -> memref<64x64xf16>
                %46 = arith.muli %arg5, %c4 : index
                %47 = arith.addi %arg4, %46 : index
                loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %47], LR : [%arg3, %47]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y4y2__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc4_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c32 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %19 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg4, %arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.muli %arg3, %c4 : index
                %31 = arith.addi %30, %c3 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %30], LR : [%c7, %31]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %32 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %33 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %35 = arith.addi %arg5, %30 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %35], LR : [%arg4, %35]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %36 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %37 = linalg.matmul ins(%32, %36 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.copy ins(%37 : tensor<64x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %42 = arith.muli %18, %c64 : index
                %43 = arith.muli %19, %c64 : index
                %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%42, %43)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<64x64xf16> -> memref<64x64xf16>
                %46 = arith.muli %arg3, %c4 : index
                %47 = arith.addi %arg5, %46 : index
                loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %47], LR : [%arg4, %47]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y8y1__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc8_dim_y_level1_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg3, %arg4)
              %19 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %20 = loom.semaphore_take %19 : memref<64x64xf16> -> memref<64x64xf16>
              %21 = loom.init_tensor %20[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %23 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
              %24 = loom.semaphore_take %23 : memref<64x512xf16> -> memref<64x512xf16>
              %25 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
              %26 = loom.semaphore_take %25 : memref<512x64xf16> -> memref<512x64xf16>
              %27 = arith.muli %18, %c64 : index
              %c0_0 = arith.constant 0 : index
              %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%27, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
              %29 = arith.muli %arg5, %c8 : index
              %30 = arith.addi %arg4, %29 : index
              loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %30], LR : [%arg3, %30]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
              %31 = loom.bufferize_to_tensor %24[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
              %32 = arith.muli %arg6, %c64 : index
              %c0_1 = arith.constant 0 : index
              %33 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %32)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_2, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
              %34 = loom.bufferize_to_tensor %26[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
              %35 = linalg.matmul ins(%31, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
              loom.semaphore_give %26 : memref<512x64xf16>
              loom.semaphore_give %24 : memref<64x512xf16>
              %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
              %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %39 = linalg.copy ins(%35 : tensor<64x64xf16>) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
              loom.semaphore_give %20 : memref<64x64xf16>
              %40 = arith.muli %18, %c64 : index
              %41 = arith.muli %arg6, %c64 : index
              %42 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%40, %41)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              %43 = loom.bufferize_to_memref %39 : tensor<64x64xf16> -> memref<64x64xf16>
              %44 = arith.muli %arg5, %c8 : index
              %45 = arith.addi %arg4, %44 : index
              loom.copy %43, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %45], LR : [%arg3, %45]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %37 : memref<64x64xf16>
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y8y1__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg4, %arg5)
              %19 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %20 = loom.semaphore_take %19 : memref<64x64xf16> -> memref<64x64xf16>
              %21 = loom.init_tensor %20[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %23 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
              %24 = loom.semaphore_take %23 : memref<64x512xf16> -> memref<64x512xf16>
              %25 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
              %26 = loom.semaphore_take %25 : memref<512x64xf16> -> memref<512x64xf16>
              %27 = arith.muli %arg6, %c64 : index
              %c0_0 = arith.constant 0 : index
              %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%27, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
              %29 = loom.bufferize_to_tensor %24[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
              %30 = arith.muli %18, %c64 : index
              %c0_1 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %30)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
              %32 = arith.muli %arg3, %c8 : index
              %33 = arith.addi %arg5, %32 : index
              loom.copy %reinterpret_cast_2, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %33], LR : [%arg4, %33]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
              %34 = loom.bufferize_to_tensor %26[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
              %35 = linalg.matmul ins(%29, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
              loom.semaphore_give %26 : memref<512x64xf16>
              loom.semaphore_give %24 : memref<64x512xf16>
              %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
              %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %39 = linalg.copy ins(%35 : tensor<64x64xf16>) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
              loom.semaphore_give %20 : memref<64x64xf16>
              %40 = arith.muli %arg6, %c64 : index
              %41 = arith.muli %18, %c64 : index
              %42 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%40, %41)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              %43 = loom.bufferize_to_memref %39 : tensor<64x64xf16> -> memref<64x64xf16>
              %44 = arith.muli %arg3, %c8 : index
              %45 = arith.addi %arg5, %44 : index
              loom.copy %43, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %45], LR : [%arg4, %45]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %37 : memref<64x64xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x1x8_y8__d0i0_d1i0_d2i1__f01__dim_x_level1_bc8_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              scf.for %arg7 = %c0 to %c8 step %c1 {
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %30 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %31 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %31)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %33 = arith.addi %arg3, %arg5 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%33, %c0], LR : [%33, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %34 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %35 = linalg.matmul ins(%30, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %39 = linalg.copy ins(%35 : tensor<64x64xf16>) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %40 = arith.muli %18, %c64 : index
                %41 = arith.muli %19, %c64 : index
                %42 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%40, %41)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %43 = loom.bufferize_to_memref %39 : tensor<64x64xf16> -> memref<64x64xf16>
                %44 = arith.addi %arg3, %arg5 : index
                loom.copy %43, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%44, %arg4], LR : [%44, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %37 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x1x8_y8__d0i1_d1i1_d2i0__f01__dim_y_level0_bc8_dim_x_level1_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (1) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              scf.for %arg7 = %c0 to %c8 step %c1 {
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.addi %arg4, %arg3 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%30, %c0], LR : [%30, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %31 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %32 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %33 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %32)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg5], LR : [%c7, %arg5]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %34 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %35 = linalg.matmul ins(%31, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
                %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %39 = linalg.copy ins(%35 : tensor<64x64xf16>) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %40 = arith.muli %18, %c64 : index
                %41 = arith.muli %19, %c64 : index
                %42 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%40, %41)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %43 = loom.bufferize_to_memref %39 : tensor<64x64xf16> -> memref<64x64xf16>
                %44 = arith.addi %arg4, %arg3 : index
                loom.copy %43, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%44, %arg5], LR : [%44, %arg5]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %37 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x2x4_y8__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc2_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            scf.for %arg6 = %c0 to %c4 step %c1 {
              scf.for %arg7 = %c0 to %c16 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg3, %arg4, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.muli %arg5, %c2 : index
                %31 = arith.addi %arg3, %30 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%31, %arg4], LR : [%31, %arg4]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %32 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %33 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %35 = arith.addi %30, %c1 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%30, %c0], LR : [%35, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %36 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %37 = linalg.matmul ins(%32, %36 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.copy ins(%37 : tensor<64x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %42 = arith.muli %18, %c64 : index
                %43 = arith.muli %19, %c64 : index
                %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%42, %43)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<64x64xf16> -> memref<64x64xf16>
                %46 = arith.muli %arg5, %c2 : index
                %47 = arith.addi %arg3, %46 : index
                loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%47, %arg4], LR : [%47, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x2x4_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc2_dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c16 step %c1 {
              scf.for %arg7 = %c0 to %c4 step %c1 {
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %19 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg4, %arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.muli %arg3, %c2 : index
                %31 = arith.addi %30, %c1 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%30, %c0], LR : [%31, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %32 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %33 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %35 = arith.addi %arg4, %30 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%35, %arg5], LR : [%35, %arg5]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %36 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %37 = linalg.matmul ins(%32, %36 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.copy ins(%37 : tensor<64x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %42 = arith.muli %18, %c64 : index
                %43 = arith.muli %19, %c64 : index
                %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%42, %43)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<64x64xf16> -> memref<64x64xf16>
                %46 = arith.muli %arg3, %c2 : index
                %47 = arith.addi %arg4, %46 : index
                loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%47, %arg5], LR : [%47, %arg5]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x4x2_y8__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc4_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              scf.for %arg7 = %c0 to %c32 step %c1 {
                %18 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg3, %arg4, %arg6)
                %19 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.muli %arg5, %c4 : index
                %31 = arith.addi %arg3, %30 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%31, %arg4], LR : [%31, %arg4]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %32 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %33 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %35 = arith.addi %30, %c3 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%30, %c0], LR : [%35, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %36 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %37 = linalg.matmul ins(%32, %36 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.copy ins(%37 : tensor<64x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %42 = arith.muli %18, %c64 : index
                %43 = arith.muli %19, %c64 : index
                %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%42, %43)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<64x64xf16> -> memref<64x64xf16>
                %46 = arith.muli %arg5, %c4 : index
                %47 = arith.addi %arg3, %46 : index
                loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%47, %arg4], LR : [%47, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x4x2_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc4_dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c32 step %c1 {
              scf.for %arg7 = %c0 to %c2 step %c1 {
                %18 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %19 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg4, %arg5, %arg7)
                %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
                %22 = loom.init_tensor %21[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
                %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
                %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
                %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
                %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
                %28 = arith.muli %18, %c64 : index
                %c0_0 = arith.constant 0 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%28, %c0_0)
                %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
                %30 = arith.muli %arg3, %c4 : index
                %31 = arith.addi %30, %c3 : index
                loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%30, %c0], LR : [%31, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
                %32 = loom.bufferize_to_tensor %25[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
                %33 = arith.muli %19, %c64 : index
                %c0_1 = arith.constant 0 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %33)
                %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
                %35 = arith.addi %arg4, %30 : index
                loom.copy %reinterpret_cast_2, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%35, %arg5], LR : [%35, %arg5]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
                %36 = loom.bufferize_to_tensor %27[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
                %37 = linalg.matmul ins(%32, %36 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%23 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %27 : memref<512x64xf16>
                loom.semaphore_give %25 : memref<64x512xf16>
                %38 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %39 = loom.semaphore_take %38 : memref<64x64xf16> -> memref<64x64xf16>
                %40 = loom.init_tensor %39[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %41 = linalg.copy ins(%37 : tensor<64x64xf16>) outs(%40 : tensor<64x64xf16>) -> tensor<64x64xf16>
                loom.semaphore_give %21 : memref<64x64xf16>
                %42 = arith.muli %18, %c64 : index
                %43 = arith.muli %19, %c64 : index
                %44 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%42, %43)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%44], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<64x64xf16> -> memref<64x64xf16>
                %46 = arith.muli %arg3, %c4 : index
                %47 = arith.addi %arg4, %46 : index
                loom.copy %45, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%47, %arg5], LR : [%47, %arg5]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<64x64xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8x1_y8__d0i0_d1i0_d2i1__f01__n_dim_x_level1_bc8_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg3, %arg4)
              %19 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %20 = loom.semaphore_take %19 : memref<64x64xf16> -> memref<64x64xf16>
              %21 = loom.init_tensor %20[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %23 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
              %24 = loom.semaphore_take %23 : memref<64x512xf16> -> memref<64x512xf16>
              %25 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
              %26 = loom.semaphore_take %25 : memref<512x64xf16> -> memref<512x64xf16>
              %27 = arith.muli %18, %c64 : index
              %c0_0 = arith.constant 0 : index
              %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%27, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
              %29 = arith.muli %arg5, %c8 : index
              %30 = arith.addi %arg3, %29 : index
              loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%30, %arg4], LR : [%30, %arg4]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
              %31 = loom.bufferize_to_tensor %24[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
              %32 = arith.muli %arg6, %c64 : index
              %c0_1 = arith.constant 0 : index
              %33 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %32)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_2, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
              %34 = loom.bufferize_to_tensor %26[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
              %35 = linalg.matmul ins(%31, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
              loom.semaphore_give %26 : memref<512x64xf16>
              loom.semaphore_give %24 : memref<64x512xf16>
              %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
              %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %39 = linalg.copy ins(%35 : tensor<64x64xf16>) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
              loom.semaphore_give %20 : memref<64x64xf16>
              %40 = arith.muli %18, %c64 : index
              %41 = arith.muli %arg6, %c64 : index
              %42 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%40, %41)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              %43 = loom.bufferize_to_memref %39 : tensor<64x64xf16> -> memref<64x64xf16>
              %44 = arith.muli %arg5, %c8 : index
              %45 = arith.addi %arg3, %44 : index
              loom.copy %43, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%45, %arg4], LR : [%45, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %37 : memref<64x64xf16>
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8x1_y8__d0i1_d1i1_d2i0__f01__dim_x_level1_bc8_dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %c64 step %c1 {
              %18 = affine.apply affine_map<(d0, d1) -> (d0 * 8 + d1)>(%arg4, %arg5)
              %19 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %20 = loom.semaphore_take %19 : memref<64x64xf16> -> memref<64x64xf16>
              %21 = loom.init_tensor %20[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<64x64xf16>) -> tensor<64x64xf16>
              %23 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
              %24 = loom.semaphore_take %23 : memref<64x512xf16> -> memref<64x512xf16>
              %25 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
              %26 = loom.semaphore_take %25 : memref<512x64xf16> -> memref<512x64xf16>
              %27 = arith.muli %arg6, %c64 : index
              %c0_0 = arith.constant 0 : index
              %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%27, %c0_0)
              %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
              %29 = loom.bufferize_to_tensor %24[64, 512] : memref<64x512xf16> -> tensor<64x512xf16>
              %30 = arith.muli %18, %c64 : index
              %c0_1 = arith.constant 0 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%c0_1, %30)
              %reinterpret_cast_2 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
              %32 = arith.muli %arg3, %c8 : index
              %33 = arith.addi %arg4, %32 : index
              loom.copy %reinterpret_cast_2, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%33, %arg5], LR : [%33, %arg5]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
              %34 = loom.bufferize_to_tensor %26[512, 64] : memref<512x64xf16> -> tensor<512x64xf16>
              %35 = linalg.matmul ins(%29, %34 : tensor<64x512xf16>, tensor<512x64xf16>) outs(%22 : tensor<64x64xf16>) -> tensor<64x64xf16>
              loom.semaphore_give %26 : memref<512x64xf16>
              loom.semaphore_give %24 : memref<64x512xf16>
              %36 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
              %37 = loom.semaphore_take %36 : memref<64x64xf16> -> memref<64x64xf16>
              %38 = loom.init_tensor %37[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
              %39 = linalg.copy ins(%35 : tensor<64x64xf16>) outs(%38 : tensor<64x64xf16>) -> tensor<64x64xf16>
              loom.semaphore_give %20 : memref<64x64xf16>
              %40 = arith.muli %arg6, %c64 : index
              %41 = arith.muli %18, %c64 : index
              %42 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%40, %41)
              %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%42], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              %43 = loom.bufferize_to_memref %39 : tensor<64x64xf16> -> memref<64x64xf16>
              %44 = arith.muli %arg3, %c8 : index
              %45 = arith.addi %arg4, %44 : index
              loom.copy %43, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%45, %arg5], LR : [%45, %arg5]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %37 : memref<64x64xf16>
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
