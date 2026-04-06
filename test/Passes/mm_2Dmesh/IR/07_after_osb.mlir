module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  %0 = adl.memory.bank "DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dram_channel", 8
  %2 = adl.memory.array "DRAM", [%1] of %0
  %3 = adl.memory.bank "bank", {bsize = 16 : i64, nblk = 5856 : i64}
  %4 = adl.spatial_dim "nbank", 16
  %5 = adl.memory.array "L1", [%4] of %3
  %6 = adl.processor.compute @matrix_lane, [(%5, %5)]
  %7 = adl.processor.compute @vector_lane, [(%5, %5)]
  %8 = adl.arch.compose "core", arch[%6, %7], mem[%5]
  %9 = adl.spatial_dim "x", 8
  %10 = adl.spatial_dim "y", 8
  %11 = adl.arch.scale "mesh", [%9, %10] of %8
  %12 = adl.processor.dmover @dram_l1_mover, [(%2, %5), (%5, %2)]
  %13 = adl.arch.compose "system", arch[%11, %12], mem[%2]
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg5, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg6, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg6, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg5, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg6, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg6, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg5, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg6, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg6, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_a_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg5, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg6, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg6, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg6, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg5, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg5, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg6, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg5, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg5, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg6, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg5, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg5, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_a_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg6, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg5, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg5, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg5, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg6, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg6, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg5, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg6, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg6, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg5, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg6, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg6, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_a_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg5, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg6, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg6, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg6, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg5, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg5, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg6, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg5, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg5, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg6, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg5, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg5, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_a_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c64 step %c1 {
          scf.for %arg6 = %c0 to %c64 step %c1 {
            %14 = arith.muli %arg3, %c8 overflow<nsw> : index
            %15 = arith.addi %14, %arg4 : index
            %16 = arith.muli %arg6, %c64 overflow<nsw> : index
            %17 = arith.addi %15, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %17, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %arg5, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %arg5, %c64 : index
            %25 = arith.muli %17, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg6, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg6, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg6, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__x_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg6, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg5, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg5, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg5, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__x_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg5, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg6, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg6, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg6, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__y_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c512 step %c1 {
          scf.for %arg6 = %c0 to %c8 step %c1 {
            %14 = arith.muli %arg5, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg6, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg5, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg5, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg5, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__y_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c8 step %c1 {
          scf.for %arg6 = %c0 to %c512 step %c1 {
            %14 = arith.muli %arg6, %c8 overflow<nsw> : index
            %15 = arith.addi %arg3, %14 : index
            %16 = arith.muli %arg5, %c8 overflow<nsw> : index
            %17 = arith.addi %arg4, %16 : index
            %18 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
            %19 = loom.semaphore_take %18 : memref<128x64xf16> -> memref<128x64xf16>
            %20 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
            %21 = loom.semaphore_take %20 : memref<1x128xf16> -> memref<1x128xf16>
            %22 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %23 = loom.semaphore_take %22 : memref<1x64xf16> -> memref<1x64xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x64xf16>)
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %27 = arith.muli %arg7, %c128 : index
              %28 = arith.muli %15, %c512 overflow<nsw> : index
              %29 = arith.addi %28, %27 : index
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %21 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
              %30 = arith.muli %17, %c64 : index
              %31 = arith.muli %arg7, %c524288 : index
              %32 = arith.addi %31, %30 : index
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
              linalg.matmul ins(%21, %19 : memref<1x128xf16>, memref<128x64xf16>) outs(%23 : memref<1x64xf16>)
              loom.semaphore_give %19 : memref<128x64xf16>
              loom.semaphore_give %21 : memref<1x128xf16>
            }
            %cast = memref.cast %23 : memref<1x64xf16> to memref<?x?xf16>
            %24 = arith.muli %17, %c64 : index
            %25 = arith.muli %15, %c4096 overflow<nsw> : index
            %26 = arith.addi %25, %24 : index
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x64xf16>
          }
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__a_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__a_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@x, @y]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__a_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__a_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c524288 = arith.constant 524288 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c4096 = arith.constant 4096 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg5 = %c0 to %c4096 step %c1 {
          %14 = arith.muli %arg3, %c8 overflow<nsw> : index
          %15 = arith.addi %14, %arg4 : index
          %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
          %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
          %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
          %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
          %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
          %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
          linalg.fill ins(%cst : f16) outs(%21 : memref<1x64xf16>)
          scf.for %arg6 = %c0 to %c4 step %c1 {
            %25 = arith.muli %arg6, %c128 : index
            %26 = arith.muli %arg5, %c512 overflow<nsw> : index
            %27 = arith.addi %26, %25 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%27], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
            %28 = arith.muli %15, %c64 : index
            %29 = arith.muli %arg6, %c524288 : index
            %30 = arith.addi %29, %28 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%30], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
            linalg.matmul ins(%19, %17 : memref<1x128xf16>, memref<128x64xf16>) outs(%21 : memref<1x64xf16>)
            loom.semaphore_give %17 : memref<128x64xf16>
            loom.semaphore_give %19 : memref<1x128xf16>
          }
          %cast = memref.cast %21 : memref<1x64xf16> to memref<?x?xf16>
          %22 = arith.muli %15, %c64 : index
          %23 = arith.muli %arg5, %c4096 overflow<nsw> : index
          %24 = arith.addi %23, %22 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%24], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.copy %cast, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %21 : memref<1x64xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.mapped_to_dims = [@y, @x]}
      return
    }
  }
}
