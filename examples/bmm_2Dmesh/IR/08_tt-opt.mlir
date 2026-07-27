module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f012__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f012__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f012__dim_y_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f012__dim_y_level0_bc2_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f021__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f021__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f021__dim_y_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f021__dim_y_level0_bc2_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f102__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f102__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f102__dim_y_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f102__dim_y_level0_bc2_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f120__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f120__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f120__dim_y_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f120__dim_y_level0_bc2_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f201__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f201__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f201__dim_y_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f201__dim_y_level0_bc2_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f210__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f210__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f210__dim_y_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d0i1_d1i2__f210__dim_y_level0_bc2_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__n_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f012__dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f021__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f021__n_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f021__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f021__dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f102__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f102__n_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f102__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f102__dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f120__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f120__n_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f120__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f120__dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f201__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f201__n_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f201__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f201__dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f210__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f210__n_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f210__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y2y4__d2i0_d1i1_d0i2__f210__dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f012__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f012__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f012__dim_y_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f012__dim_y_level0_bc4_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f021__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f021__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f021__dim_y_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f021__dim_y_level0_bc4_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f102__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f102__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f102__dim_y_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f102__dim_y_level0_bc4_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f120__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f120__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f120__dim_y_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f120__dim_y_level0_bc4_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f201__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f201__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f201__dim_y_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f201__dim_y_level0_bc4_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f210__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f210__n_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f210__dim_y_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d0i1_d1i2__f210__dim_y_level0_bc4_dim_x_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__n_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f012__dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f021__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f021__n_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f021__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f021__dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f102__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f102__n_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f102__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f102__dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f120__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f120__n_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f120__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f120__dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f201__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f201__n_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f201__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f201__dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f210__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f210__n_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f210__dim_x_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x8_y4y2__d2i0_d1i1_d0i2__f210__dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f012__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f012__n_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f012__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f012__dim_y_level0_bc8_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f021__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f021__n_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f021__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f021__dim_y_level0_bc8_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f102__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f102__n_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f102__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f102__dim_y_level0_bc8_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f120__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f120__n_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f120__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f120__dim_y_level0_bc8_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f201__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f201__n_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f201__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f201__dim_y_level0_bc8_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f210__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f210__n_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f210__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d0i1_d1i2__f210__dim_y_level0_bc8_dim_x_level0_bc2_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg7, %c2 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__dim_x_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f012__dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f021__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f021__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f021__dim_x_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f021__dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f102__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f102__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f102__dim_x_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f102__dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f120__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f120__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f120__dim_x_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f120__dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c32 = arith.constant 32 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c32 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f201__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f201__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f201__dim_x_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f201__dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f210__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f210__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f210__dim_x_level0_bc2_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x2x4_y8__d2i0_d1i1_d0i2__f210__dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      %c2 = arith.constant 2 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c2 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f012__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f012__n_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f012__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f012__dim_y_level0_bc8_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f021__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f021__n_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f021__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f021__dim_y_level0_bc8_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f102__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f102__n_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f102__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f102__dim_y_level0_bc8_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f120__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f120__n_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f120__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f120__dim_y_level0_bc8_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg6, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f201__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f201__n_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f201__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f201__dim_y_level0_bc8_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f210__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f210__n_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f210__dim_y_level0_bc8_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d0i1_d1i2__f210__dim_y_level0_bc8_dim_x_level0_bc4_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg7, %c4 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__dim_x_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f012__dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f021__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f021__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f021__dim_x_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f021__dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f102__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f102__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f102__dim_x_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f102__dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f120__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f120__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f120__dim_x_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f120__dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c8 step %c1 {
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg7, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f201__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f201__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f201__dim_x_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f201__dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f210__n_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f210__n_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f210__dim_x_level0_bc4_n_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @batch_matmul__x4x2_y8__d2i0_d1i1_d0i2__f210__dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b64__tile_k64__tile_m64__tile_n64(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
      %c1073741824 = arith.constant 1073741824 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c134217728 = arith.constant 134217728 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg3, %arg4, %arg5) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg6 = %c0 to %c16 step %c1 {
          scf.for %arg7 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg7, %c8 overflow<nsw> : index
            %21 = arith.addi %arg4, %20 : index
            %22 = arith.muli %arg6, %c4 overflow<nsw> : index
            %23 = arith.addi %arg5, %22 : index
            %24 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %25 = loom.semaphore_take %24 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %26 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %27 = loom.semaphore_take %26 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            %28 = loom.alloc [64, 64, 64] on @L1 : memref<64x64x64xf16>
            %29 = loom.semaphore_take %28 : memref<64x64x64xf16> -> memref<64x64x64xf16>
            scf.for %arg8 = %c0 to %c8 step %c1 {
              %35 = arith.muli %arg8, %c64 : index
              %36 = arith.muli %arg3, %c134217728 : index
              %37 = arith.muli %21, %c32768 : index
              %38 = arith.addi %36, %37 : index
              %39 = arith.addi %38, %35 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%39], sizes: [64, 64, 64], strides: [2097152, 512, 1] : memref<8x4096x512xf16> to memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] : memref<64x64x64xf16, strided<[2097152, 512, 1], offset: ?>> to memref<64x64x64xf16>
              %40 = arith.muli %23, %c64 : index
              %41 = arith.muli %arg8, %c262144 : index
              %42 = arith.addi %36, %41 : index
              %43 = arith.addi %42, %40 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%43], sizes: [64, 64, 64], strides: [2097152, 4096, 1] : memref<8x512x4096xf16> to memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<64x64x64xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<64x64x64xf16>
              loom.batch_matmul ins(%27, %29 : memref<64x64x64xf16>, memref<64x64x64xf16>) outs(%25 : memref<64x64x64xf16>)
              loom.semaphore_give %29 : memref<64x64x64xf16>
              loom.semaphore_give %27 : memref<64x64x64xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %arg3, %c1073741824 : index
            %32 = arith.muli %21, %c262144 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64, 64], strides: [16777216, 4096, 1] : memref<8x4096x4096xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<64x64x64xf16> to memref<64x64x64xf16, strided<[16777216, 4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<64x64x64xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_y, @dim_x]}
      return
    }
  }
}
