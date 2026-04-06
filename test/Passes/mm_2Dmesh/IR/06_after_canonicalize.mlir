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
    func.func @_matmul__d0i0_d1i0__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg6, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg6, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_dim_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg6, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg6, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_dim_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg6, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg6, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_a_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg6, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg6, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg5, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg5, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_dim_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg5, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg5, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_dim_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg5, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg5, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_a_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg5, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg5, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg6, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg6, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_dim_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg6, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg6, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_dim_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg6, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg6, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_a_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg6, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg6, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg5, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg5, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_dim_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg5, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg5, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_dim_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg5, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg5, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_a_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %arg5, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %arg5, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__n_dim_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__dim_x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__dim_x_dim_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__n_dim_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__dim_x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__dim_x_dim_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__n_dim_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__dim_y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__dim_y_dim_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__n_dim_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__dim_y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__dim_y_dim_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %22 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %23 = loom.semaphore_take %22 : memref<128x64xf16> -> memref<128x64xf16>
              %24 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %25 = loom.semaphore_take %24 : memref<1x128xf16> -> memref<1x128xf16>
              %26 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %27 = loom.semaphore_take %26 : memref<1x64xf16> -> memref<1x64xf16>
              %28 = loom.init_tensor %27[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %30 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %29) -> (tensor<1x64xf16>) {
                %34 = arith.muli %arg7, %c128 : index
                %35 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%20, %34)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%35], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %36 = loom.bufferize_to_tensor %25[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %37 = arith.muli %21, %c64 : index
                %38 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%34, %37)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%38], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %39 = loom.bufferize_to_tensor %23[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %40 = linalg.matmul ins(%36, %39 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %23 : memref<128x64xf16>
                loom.semaphore_give %25 : memref<1x128xf16>
                affine.yield %40 : tensor<1x64xf16>
              }
              %31 = arith.muli %21, %c64 : index
              %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %31)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %33 = loom.bufferize_to_memref %30 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %33, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__dim_y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__dim_x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__a_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__dim_y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__dim_x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__a_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__dim_y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__dim_x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__a_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__n_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__dim_y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__dim_x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__a_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %20 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %21 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %22 = loom.semaphore_take %21 : memref<128x64xf16> -> memref<128x64xf16>
              %23 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %24 = loom.semaphore_take %23 : memref<1x128xf16> -> memref<1x128xf16>
              %25 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %26 = loom.semaphore_take %25 : memref<1x64xf16> -> memref<1x64xf16>
              %27 = loom.init_tensor %26[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %29 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %28) -> (tensor<1x64xf16>) {
                %33 = arith.muli %arg7, %c128 : index
                %34 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %33)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%34], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %35 = loom.bufferize_to_tensor %24[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %36 = arith.muli %20, %c64 : index
                %37 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%33, %36)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%37], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %38 = loom.bufferize_to_tensor %22[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %39 = linalg.matmul ins(%35, %38 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %22 : memref<128x64xf16>
                loom.semaphore_give %24 : memref<1x128xf16>
                affine.yield %39 : tensor<1x64xf16>
              }
              %30 = arith.muli %20, %c64 : index
              %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %30)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %32 = loom.bufferize_to_memref %29 : tensor<1x64xf16> -> memref<1x64xf16>
              loom.copy %32, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<1x64xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %26 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @dim_y}
      return
    }
  }
}
