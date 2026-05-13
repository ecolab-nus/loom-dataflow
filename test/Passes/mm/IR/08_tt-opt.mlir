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
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %18 = arith.muli %arg5, %c8 overflow<nsw> : index
            %19 = arith.addi %arg3, %18 : index
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %23 = loom.semaphore_take %22 : memref<64x64xf16> -> memref<64x64xf16>
            %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
            %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
            %28 = arith.muli %19, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg3, %c0], LR : [%arg3, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %29 = arith.muli %21, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%25, %27 : memref<64x512xf16>, memref<512x64xf16>) outs(%23 : memref<64x64xf16>)
            loom.semaphore_give %27 : memref<512x64xf16>
            loom.semaphore_give %25 : memref<64x512xf16>
            %30 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %31 = loom.semaphore_take %30 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %23, %31 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %23 : memref<64x64xf16>
            %32 = arith.muli %19, %c262144 : index
            %33 = arith.addi %32, %29 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %31, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %31 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y1y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %18 = arith.muli %arg5, %c8 overflow<nsw> : index
            %19 = arith.addi %arg3, %18 : index
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %23 = loom.semaphore_take %22 : memref<64x64xf16> -> memref<64x64xf16>
            %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
            %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
            %28 = arith.muli %19, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg3], LR : [%c7, %arg3]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %29 = arith.muli %21, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%25, %27 : memref<64x512xf16>, memref<512x64xf16>) outs(%23 : memref<64x64xf16>)
            loom.semaphore_give %27 : memref<512x64xf16>
            loom.semaphore_give %25 : memref<64x512xf16>
            %30 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %31 = loom.semaphore_take %30 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %23, %31 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %23 : memref<64x64xf16>
            %32 = arith.muli %19, %c262144 : index
            %33 = arith.addi %32, %29 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %31, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %arg3], LR : [%arg4, %arg3]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %31 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y2y4__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc8_dim_y_level0_bc2_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c8, %c2, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %18 = arith.muli %arg3, %c8 overflow<nsw> : index
            %19 = arith.addi %18, %arg4 : index
            %20 = arith.muli %arg6, %c16 overflow<nsw> : index
            %21 = arith.addi %19, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64xf16> -> memref<64x64xf16>
            %26 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %27 = loom.semaphore_take %26 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %29 = loom.semaphore_take %28 : memref<512x64xf16> -> memref<512x64xf16>
            %30 = arith.muli %21, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            %31 = arith.muli %arg5, %c2 : index
            %32 = arith.addi %arg4, %31 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %32], LR : [%arg3, %32]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %33 = arith.muli %23, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            %34 = arith.addi %31, %c1 : index
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %31], LR : [%c7, %34]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%27, %29 : memref<64x512xf16>, memref<512x64xf16>) outs(%25 : memref<64x64xf16>)
            loom.semaphore_give %29 : memref<512x64xf16>
            loom.semaphore_give %27 : memref<64x512xf16>
            %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %25, %36 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %25 : memref<64x64xf16>
            %37 = arith.muli %21, %c262144 : index
            %38 = arith.addi %37, %33 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %36, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %32], LR : [%arg3, %32]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %36 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_m, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0, 1], loom.physical_dims = [@dim_x, @dim_y, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y2y4__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc2_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %18 = arith.muli %arg6, %c4 overflow<nsw> : index
            %19 = arith.addi %arg3, %18 : index
            %20 = arith.muli %arg4, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg5 : index
            %22 = arith.muli %arg7, %c16 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64xf16> -> memref<64x64xf16>
            %26 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %27 = loom.semaphore_take %26 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %29 = loom.semaphore_take %28 : memref<512x64xf16> -> memref<512x64xf16>
            %30 = arith.muli %19, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            %31 = arith.muli %arg3, %c2 : index
            %32 = arith.addi %31, %c1 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 2] region : (UL : [%c0, %31], LR : [%c7, %32]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %33 = arith.muli %23, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            %34 = arith.addi %arg5, %31 : index
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %34], LR : [%arg4, %34]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%27, %29 : memref<64x512xf16>, memref<512x64xf16>) outs(%25 : memref<64x64xf16>)
            loom.semaphore_give %29 : memref<512x64xf16>
            loom.semaphore_give %27 : memref<64x512xf16>
            %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %25, %36 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %25 : memref<64x64xf16>
            %37 = arith.muli %19, %c262144 : index
            %38 = arith.addi %37, %33 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %36, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %34], LR : [%arg4, %34]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %36 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_n, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y4y2__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc8_dim_y_level0_bc4_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c8, %c4, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %18 = arith.muli %arg3, %c8 overflow<nsw> : index
            %19 = arith.addi %18, %arg4 : index
            %20 = arith.muli %arg6, %c32 overflow<nsw> : index
            %21 = arith.addi %19, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64xf16> -> memref<64x64xf16>
            %26 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %27 = loom.semaphore_take %26 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %29 = loom.semaphore_take %28 : memref<512x64xf16> -> memref<512x64xf16>
            %30 = arith.muli %21, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            %31 = arith.muli %arg5, %c4 : index
            %32 = arith.addi %arg4, %31 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %32], LR : [%arg3, %32]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %33 = arith.muli %23, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            %34 = arith.addi %31, %c3 : index
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %31], LR : [%c7, %34]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%27, %29 : memref<64x512xf16>, memref<512x64xf16>) outs(%25 : memref<64x64xf16>)
            loom.semaphore_give %29 : memref<512x64xf16>
            loom.semaphore_give %27 : memref<64x512xf16>
            %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %25, %36 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %25 : memref<64x64xf16>
            %37 = arith.muli %21, %c262144 : index
            %38 = arith.addi %37, %33 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %36, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %32], LR : [%arg3, %32]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %36 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_m, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0, 1], loom.physical_dims = [@dim_x, @dim_y, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y4y2__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level0_bc4_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %18 = arith.muli %arg6, %c2 overflow<nsw> : index
            %19 = arith.addi %arg3, %18 : index
            %20 = arith.muli %arg4, %c8 overflow<nsw> : index
            %21 = arith.addi %20, %arg5 : index
            %22 = arith.muli %arg7, %c32 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64xf16> -> memref<64x64xf16>
            %26 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %27 = loom.semaphore_take %26 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %29 = loom.semaphore_take %28 : memref<512x64xf16> -> memref<512x64xf16>
            %30 = arith.muli %19, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            %31 = arith.muli %arg3, %c4 : index
            %32 = arith.addi %31, %c3 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 4] region : (UL : [%c0, %31], LR : [%c7, %32]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %33 = arith.muli %23, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            %34 = arith.addi %arg5, %31 : index
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg4, %34], LR : [%arg4, %34]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%27, %29 : memref<64x512xf16>, memref<512x64xf16>) outs(%25 : memref<64x64xf16>)
            loom.semaphore_give %29 : memref<512x64xf16>
            loom.semaphore_give %27 : memref<64x512xf16>
            %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %25, %36 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %25 : memref<64x64xf16>
            %37 = arith.muli %19, %c262144 : index
            %38 = arith.addi %37, %33 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %36, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %34], LR : [%arg4, %34]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %36 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_n, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y8y1__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc8_dim_y_level1_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %18 = arith.muli %arg3, %c8 overflow<nsw> : index
          %19 = arith.addi %18, %arg4 : index
          %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
          %22 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %23 = loom.semaphore_take %22 : memref<64x512xf16> -> memref<64x512xf16>
          %24 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
          %25 = loom.semaphore_take %24 : memref<512x64xf16> -> memref<512x64xf16>
          %26 = arith.muli %19, %c32768 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
          %27 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
          linalg.matmul ins(%23, %25 : memref<64x512xf16>, memref<512x64xf16>) outs(%21 : memref<64x64xf16>)
          loom.semaphore_give %25 : memref<512x64xf16>
          loom.semaphore_give %23 : memref<64x512xf16>
          %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %29 = loom.semaphore_take %28 : memref<64x64xf16> -> memref<64x64xf16>
          loom.copy %21, %29 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
          loom.semaphore_give %21 : memref<64x64xf16>
          %30 = arith.muli %19, %c262144 : index
          %31 = arith.addi %30, %27 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %29, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %29 : memref<64x64xf16>
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y8y1__d0i1_d1i1_d2i0__f01__dim_x_level0_bc8_dim_y_level1_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %18 = arith.muli %arg3, %c8 overflow<nsw> : index
          %19 = arith.addi %18, %arg4 : index
          %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
          %22 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %23 = loom.semaphore_take %22 : memref<64x512xf16> -> memref<64x512xf16>
          %24 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
          %25 = loom.semaphore_take %24 : memref<512x64xf16> -> memref<512x64xf16>
          %26 = arith.muli %arg5, %c32768 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
          %27 = arith.muli %19, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
          linalg.matmul ins(%23, %25 : memref<64x512xf16>, memref<512x64xf16>) outs(%21 : memref<64x64xf16>)
          loom.semaphore_give %25 : memref<512x64xf16>
          loom.semaphore_give %23 : memref<64x512xf16>
          %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %29 = loom.semaphore_take %28 : memref<64x64xf16> -> memref<64x64xf16>
          loom.copy %21, %29 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
          loom.semaphore_give %21 : memref<64x64xf16>
          %30 = arith.muli %arg5, %c262144 : index
          %31 = arith.addi %30, %27 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %29, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %29 : memref<64x64xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_n, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x1x8_y8__d0i0_d1i0_d2i1__f01__dim_x_level1_bc8_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %18 = arith.muli %arg5, %c8 overflow<nsw> : index
            %19 = arith.addi %arg3, %18 : index
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %23 = loom.semaphore_take %22 : memref<64x64xf16> -> memref<64x64xf16>
            %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
            %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
            %28 = arith.muli %19, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg3], LR : [%c7, %arg3]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %29 = arith.muli %21, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%25, %27 : memref<64x512xf16>, memref<512x64xf16>) outs(%23 : memref<64x64xf16>)
            loom.semaphore_give %27 : memref<512x64xf16>
            loom.semaphore_give %25 : memref<64x512xf16>
            %30 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %31 = loom.semaphore_take %30 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %23, %31 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %23 : memref<64x64xf16>
            %32 = arith.muli %19, %c262144 : index
            %33 = arith.addi %32, %29 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %31, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg4, %arg3], LR : [%arg4, %arg3]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %31 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 1], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x1x8_y8__d0i1_d1i1_d2i0__f01__dim_y_level0_bc8_dim_x_level1_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %18 = arith.muli %arg5, %c8 overflow<nsw> : index
            %19 = arith.addi %arg3, %18 : index
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %23 = loom.semaphore_take %22 : memref<64x64xf16> -> memref<64x64xf16>
            %24 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %25 = loom.semaphore_take %24 : memref<64x512xf16> -> memref<64x512xf16>
            %26 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %27 = loom.semaphore_take %26 : memref<512x64xf16> -> memref<512x64xf16>
            %28 = arith.muli %19, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 8] region : (UL : [%arg3, %c0], LR : [%arg3, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %29 = arith.muli %21, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%29], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%25, %27 : memref<64x512xf16>, memref<512x64xf16>) outs(%23 : memref<64x64xf16>)
            loom.semaphore_give %27 : memref<512x64xf16>
            loom.semaphore_give %25 : memref<64x512xf16>
            %30 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %31 = loom.semaphore_take %30 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %23, %31 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %23 : memref<64x64xf16>
            %32 = arith.muli %19, %c262144 : index
            %33 = arith.addi %32, %29 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %31, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %31 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x2x4_y8__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc2_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %18 = arith.muli %arg3, %c2 overflow<nsw> : index
            %19 = arith.addi %18, %arg4 : index
            %20 = arith.muli %arg6, %c16 overflow<nsw> : index
            %21 = arith.addi %19, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64xf16> -> memref<64x64xf16>
            %26 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %27 = loom.semaphore_take %26 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %29 = loom.semaphore_take %28 : memref<512x64xf16> -> memref<512x64xf16>
            %30 = arith.muli %21, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            %31 = arith.muli %arg5, %c2 : index
            %32 = arith.addi %arg3, %31 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%32, %arg4], LR : [%32, %arg4]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %33 = arith.muli %23, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            %34 = arith.addi %31, %c1 : index
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%31, %c0], LR : [%34, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%27, %29 : memref<64x512xf16>, memref<512x64xf16>) outs(%25 : memref<64x64xf16>)
            loom.semaphore_give %29 : memref<512x64xf16>
            loom.semaphore_give %27 : memref<64x512xf16>
            %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %25, %36 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %25 : memref<64x64xf16>
            %37 = arith.muli %21, %c262144 : index
            %38 = arith.addi %37, %33 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %36, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%32, %arg4], LR : [%32, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %36 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_m, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0, 1], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x2x4_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc2_dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %18 = arith.muli %arg6, %c4 overflow<nsw> : index
            %19 = arith.addi %arg3, %18 : index
            %20 = arith.muli %arg4, %c2 overflow<nsw> : index
            %21 = arith.addi %20, %arg5 : index
            %22 = arith.muli %arg7, %c16 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64xf16> -> memref<64x64xf16>
            %26 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %27 = loom.semaphore_take %26 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %29 = loom.semaphore_take %28 : memref<512x64xf16> -> memref<512x64xf16>
            %30 = arith.muli %19, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            %31 = arith.muli %arg3, %c2 : index
            %32 = arith.addi %31, %c1 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [2, 8] region : (UL : [%31, %c0], LR : [%32, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %33 = arith.muli %23, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            %34 = arith.addi %arg4, %31 : index
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%34, %arg5], LR : [%34, %arg5]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%27, %29 : memref<64x512xf16>, memref<512x64xf16>) outs(%25 : memref<64x64xf16>)
            loom.semaphore_give %29 : memref<512x64xf16>
            loom.semaphore_give %27 : memref<64x512xf16>
            %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %25, %36 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %25 : memref<64x64xf16>
            %37 = arith.muli %19, %c262144 : index
            %38 = arith.addi %37, %33 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %36, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%34, %arg5], LR : [%34, %arg5]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %36 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_n, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x4x2_y8__d0i0_d1i0_d2i1__f01__n_dim_x_level0_bc4_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c2 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %18 = arith.muli %arg3, %c4 overflow<nsw> : index
            %19 = arith.addi %18, %arg4 : index
            %20 = arith.muli %arg6, %c32 overflow<nsw> : index
            %21 = arith.addi %19, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64xf16> -> memref<64x64xf16>
            %26 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %27 = loom.semaphore_take %26 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %29 = loom.semaphore_take %28 : memref<512x64xf16> -> memref<512x64xf16>
            %30 = arith.muli %21, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            %31 = arith.muli %arg5, %c4 : index
            %32 = arith.addi %arg3, %31 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%32, %arg4], LR : [%32, %arg4]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %33 = arith.muli %23, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            %34 = arith.addi %31, %c3 : index
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%31, %c0], LR : [%34, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%27, %29 : memref<64x512xf16>, memref<512x64xf16>) outs(%25 : memref<64x64xf16>)
            loom.semaphore_give %29 : memref<512x64xf16>
            loom.semaphore_give %27 : memref<64x512xf16>
            %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %25, %36 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %25 : memref<64x64xf16>
            %37 = arith.muli %21, %c262144 : index
            %38 = arith.addi %37, %33 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %36, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%32, %arg4], LR : [%32, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %36 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_m, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0, 1], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x4x2_y8__d0i1_d1i1_d2i0__f01__dim_x_level0_bc4_dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c2 step %c1 {
            %18 = arith.muli %arg6, %c2 overflow<nsw> : index
            %19 = arith.addi %arg3, %18 : index
            %20 = arith.muli %arg4, %c4 overflow<nsw> : index
            %21 = arith.addi %20, %arg5 : index
            %22 = arith.muli %arg7, %c32 overflow<nsw> : index
            %23 = arith.addi %21, %22 : index
            %24 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64xf16> -> memref<64x64xf16>
            %26 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
            %27 = loom.semaphore_take %26 : memref<64x512xf16> -> memref<64x512xf16>
            %28 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
            %29 = loom.semaphore_take %28 : memref<512x64xf16> -> memref<512x64xf16>
            %30 = arith.muli %19, %c32768 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
            %31 = arith.muli %arg3, %c4 : index
            %32 = arith.addi %31, %c3 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [4, 8] region : (UL : [%31, %c0], LR : [%32, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
            %33 = arith.muli %23, %c64 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%33], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
            %34 = arith.addi %arg4, %31 : index
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%34, %arg5], LR : [%34, %arg5]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
            linalg.matmul ins(%27, %29 : memref<64x512xf16>, memref<512x64xf16>) outs(%25 : memref<64x64xf16>)
            loom.semaphore_give %29 : memref<512x64xf16>
            loom.semaphore_give %27 : memref<64x512xf16>
            %35 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %36 = loom.semaphore_take %35 : memref<64x64xf16> -> memref<64x64xf16>
            loom.copy %25, %36 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
            loom.semaphore_give %25 : memref<64x64xf16>
            %37 = arith.muli %19, %c262144 : index
            %38 = arith.addi %37, %33 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%38], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %36, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%34, %arg5], LR : [%34, %arg5]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %36 : memref<64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_n, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8x1_y8__d0i0_d1i0_d2i1__f01__n_dim_x_level1_bc8_dim_y_level0_bc8_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %18 = arith.muli %arg3, %c8 overflow<nsw> : index
          %19 = arith.addi %18, %arg4 : index
          %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
          %22 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %23 = loom.semaphore_take %22 : memref<64x512xf16> -> memref<64x512xf16>
          %24 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
          %25 = loom.semaphore_take %24 : memref<512x64xf16> -> memref<512x64xf16>
          %26 = arith.muli %19, %c32768 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
          %27 = arith.muli %arg5, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
          linalg.matmul ins(%23, %25 : memref<64x512xf16>, memref<512x64xf16>) outs(%21 : memref<64x64xf16>)
          loom.semaphore_give %25 : memref<512x64xf16>
          loom.semaphore_give %23 : memref<64x512xf16>
          %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %29 = loom.semaphore_take %28 : memref<64x64xf16> -> memref<64x64xf16>
          loom.copy %21, %29 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
          loom.semaphore_give %21 : memref<64x64xf16>
          %30 = arith.muli %19, %c262144 : index
          %31 = arith.addi %30, %27 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %29, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %29 : memref<64x64xf16>
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_m, @tile_m], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8x1_y8__d0i1_d1i1_d2i0__f01__dim_x_level1_bc8_dim_y_level0_bc8_n_n__tile_k512__tile_m64__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          %18 = arith.muli %arg3, %c8 overflow<nsw> : index
          %19 = arith.addi %18, %arg4 : index
          %20 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %21 = loom.semaphore_take %20 : memref<64x64xf16> -> memref<64x64xf16>
          %22 = loom.alloc [64, 512] on @L1 : memref<64x512xf16>
          %23 = loom.semaphore_take %22 : memref<64x512xf16> -> memref<64x512xf16>
          %24 = loom.alloc [512, 64] on @L1 : memref<512x64xf16>
          %25 = loom.semaphore_take %24 : memref<512x64xf16> -> memref<512x64xf16>
          %26 = arith.muli %arg5, %c32768 : index
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%26], sizes: [64, 512], strides: [512, 1] : memref<4096x512xf16> to memref<64x512xf16, strided<[512, 1], offset: ?>>
          loom.copy %reinterpret_cast, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<64x512xf16, strided<[512, 1], offset: ?>> to memref<64x512xf16>
          %27 = arith.muli %19, %c64 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%27], sizes: [512, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<512x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<512x64xf16, strided<[4096, 1], offset: ?>> to memref<512x64xf16>
          linalg.matmul ins(%23, %25 : memref<64x512xf16>, memref<512x64xf16>) outs(%21 : memref<64x64xf16>)
          loom.semaphore_give %25 : memref<512x64xf16>
          loom.semaphore_give %23 : memref<64x512xf16>
          %28 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %29 = loom.semaphore_take %28 : memref<64x64xf16> -> memref<64x64xf16>
          loom.copy %21, %29 src_mem_space @mem_L1 dst_mem_space @mem_L1, area : [1, 1] : memref<64x64xf16> to memref<64x64xf16>
          loom.semaphore_give %21 : memref<64x64xf16>
          %30 = arith.muli %arg5, %c262144 : index
          %31 = arith.addi %30, %27 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [64, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %29, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<64x64xf16> to memref<64x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %29 : memref<64x64xf16>
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.block_syms = [@tile_n, @tile_n], loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
}
