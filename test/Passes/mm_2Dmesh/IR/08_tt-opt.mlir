module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__n_dim_x_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__dim_y_level0_bc8_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__dim_y_level0_bc8_dim_x_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__n_dim_x_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__dim_y_level0_bc8_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__dim_y_level0_bc8_dim_x_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__n_dim_y_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__dim_x_level0_bc8_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__dim_x_level0_bc8_dim_y_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__n_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__n_dim_y_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__dim_x_level0_bc8_n_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__dim_x_level0_bc8_dim_y_level0_bc8_n__tile_k512__tile_m32__tile_n32(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c131072 = arith.constant 131072 : index
      %c8192 = arith.constant 8192 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c16 step %c1 {
          scf.for %arg6 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [32, 32] on @L1 : memref<32x32xf16>
            %25 = loom.semaphore_take %24 : memref<32x32xf16> -> memref<32x32xf16>
            %26 = loom.alloc [32, 512] on @L1 : memref<32x512xf16>
            %27 = loom.semaphore_take %26 : memref<32x512xf16> -> memref<32x512xf16>
            %28 = loom.alloc [512, 32] on @L1 : memref<512x32xf16>
            %29 = loom.semaphore_take %28 : memref<512x32xf16> -> memref<512x32xf16>
            %30 = arith.muli %21, %c8192 : index
            %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%30], sizes: [32, 512], strides: [256, 1] : memref<4096x256xf16> to memref<32x512xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<32x512xf16, strided<[256, 1], offset: ?>> to memref<32x512xf16>
            %31 = arith.muli %23, %c32 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [512, 32], strides: [4096, 1] : memref<256x4096xf16> to memref<512x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<512x32xf16, strided<[4096, 1], offset: ?>> to memref<512x32xf16>
            loom.matmul ins(%27, %29 : memref<32x512xf16>, memref<512x32xf16>) outs(%25 : memref<32x32xf16>)
            loom.semaphore_give %29 : memref<512x32xf16>
            loom.semaphore_give %27 : memref<32x512xf16>
            %32 = arith.muli %21, %c131072 : index
            %33 = arith.addi %32, %31 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [32, 32], strides: [4096, 1] : memref<4096x4096xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast_1 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<32x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
}
