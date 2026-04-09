module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
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
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__n_dim_x_level0_bc8_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__dim_y_level0_bc8_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__dim_y_level0_bc8_dim_x_level0_bc8_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__n_dim_x_level0_bc8_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__dim_y_level0_bc8_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f10__dim_y_level0_bc8_dim_x_level0_bc8_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__n_dim_y_level0_bc8_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__dim_x_level0_bc8_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f01__dim_x_level0_bc8_dim_y_level0_bc8_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %20 = arith.muli %arg5, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg6, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__n_dim_y_level0_bc8_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__dim_x_level0_bc8_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__x8_y8__d1i0_d0i1__f10__dim_x_level0_bc8_dim_y_level0_bc8_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %20 = arith.muli %arg6, %c8 overflow<nsw> : index
            %21 = arith.addi %arg3, %20 : index
            %22 = arith.muli %arg5, %c8 overflow<nsw> : index
            %23 = arith.addi %arg4, %22 : index
            %24 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %25 = loom.semaphore_take %24 : memref<1x64xf16> -> memref<1x64xf16>
            %26 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %27 = loom.semaphore_take %26 : memref<1x128xf16> -> memref<1x128xf16>
            %28 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %29 = loom.semaphore_take %28 : memref<128x64xf16> -> memref<128x64xf16>
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %33 = arith.muli %arg7, %c128 : index
              %34 = arith.muli %21, %c512 overflow<nsw> : index
              %35 = arith.addi %34, %33 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %36 = arith.muli %23, %c64 : index
              %37 = arith.muli %arg7, %c524288 : index
              %38 = arith.addi %37, %36 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              loom.matmul ins(%27, %29 : memref<1x128xf16>, memref<128x64xf16>) outs(%25 : memref<1x64xf16>)
              loom.semaphore_give %29 : memref<128x64xf16>
              loom.semaphore_give %27 : memref<1x128xf16>
            }
            %30 = arith.muli %23, %c64 : index
            %31 = arith.muli %21, %c4096 overflow<nsw> : index
            %32 = arith.addi %31, %30 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %25, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %25 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
}
