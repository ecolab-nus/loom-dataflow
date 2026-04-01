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
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg6, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg6, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg6, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg6, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f01__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg6, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg6, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg6, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg6, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg5, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg5, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg5, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg5, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i0__f10__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg5, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg5, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg5, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg5, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg6, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg6, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg6, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg6, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f01__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg6, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg6, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg6, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg6, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg5, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg5, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg5, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg5, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i0__f10__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 64 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg5, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg5, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %arg5, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %arg5, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
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
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f01__x_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__n_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i0_d1i1__f10__x_y_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f01__y_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 512 {
            affine.for %arg6 = 0 to 8 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
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
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__n_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i0_d0i1__f10__y_x_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 8 {
            affine.for %arg6 = 0 to 512 {
              %14 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %15 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %16 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %17 = loom.semaphore_take %16 : memref<128x64xf16> -> memref<128x64xf16>
              %18 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %19 = loom.semaphore_take %18 : memref<1x128xf16> -> memref<1x128xf16>
              %20 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %21 = loom.semaphore_take %20 : memref<1x64xf16> -> memref<1x64xf16>
              %22 = loom.init_tensor %21[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %24 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %23) -> (tensor<1x64xf16>) {
                %28 = arith.muli %arg7, %c128 : index
                %29 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%14, %28)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%29], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %19 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %30 = loom.bufferize_to_tensor %19[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %31 = arith.muli %15, %c64 : index
                %32 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%28, %31)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %17 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %33 = loom.bufferize_to_tensor %17[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %34 = linalg.matmul ins(%30, %33 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %17 : memref<128x64xf16>
                loom.semaphore_give %19 : memref<1x128xf16>
                affine.yield %34 : tensor<1x64xf16>
              }
              %cast = tensor.cast %24 : tensor<1x64xf16> to tensor<?x?xf16>
              %25 = arith.muli %15, %c64 : index
              %26 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%14, %25)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %27 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %27, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %21 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f01__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d0i1_d1i1__f10__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f01__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 4096 {
            affine.for %arg6 = 0 to 1 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg5, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg5, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__y_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index, loom.pass_name = "Materialize"} {
    func.func @_matmul__d1i1_d0i1__f10__x_n_n__tile_k128__tile_m1__tile_n64(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to 1 {
            affine.for %arg6 = 0 to 4096 {
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
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
              %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %15 = loom.alloc [128, 64] on @L1 : memref<128x64xf16>
              %16 = loom.semaphore_take %15 : memref<128x64xf16> -> memref<128x64xf16>
              %17 = loom.alloc [1, 128] on @L1 : memref<1x128xf16>
              %18 = loom.semaphore_take %17 : memref<1x128xf16> -> memref<1x128xf16>
              %19 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
              %20 = loom.semaphore_take %19 : memref<1x64xf16> -> memref<1x64xf16>
              %21 = loom.init_tensor %20[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<1x64xf16>) -> tensor<1x64xf16>
              %23 = affine.for %arg7 = 0 to 4 iter_args(%arg8 = %22) -> (tensor<1x64xf16>) {
                %27 = arith.muli %arg7, %c128 : index
                %28 = affine.apply affine_map<(d0, d1) -> (d0 * 512 + d1)>(%arg6, %27)
                %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%28], sizes: [1, 128], strides: [512, 1] : memref<4096x512xf16> to memref<1x128xf16, strided<[512, 1], offset: ?>>
                loom.copy %reinterpret_cast_0, %18 src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<1x128xf16, strided<[512, 1], offset: ?>> to memref<1x128xf16>
                %29 = loom.bufferize_to_tensor %18[1, 128] : memref<1x128xf16> -> tensor<1x128xf16>
                %30 = arith.muli %14, %c64 : index
                %31 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%27, %30)
                %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%31], sizes: [128, 64], strides: [4096, 1] : memref<512x4096xf16> to memref<128x64xf16, strided<[4096, 1], offset: ?>>
                loom.copy %reinterpret_cast_1, %16 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<128x64xf16, strided<[4096, 1], offset: ?>> to memref<128x64xf16>
                %32 = loom.bufferize_to_tensor %16[128, 64] : memref<128x64xf16> -> tensor<128x64xf16>
                %33 = linalg.matmul ins(%29, %32 : tensor<1x128xf16>, tensor<128x64xf16>) outs(%arg8 : tensor<1x64xf16>) -> tensor<1x64xf16>
                loom.semaphore_give %16 : memref<128x64xf16>
                loom.semaphore_give %18 : memref<1x128xf16>
                affine.yield %33 : tensor<1x64xf16>
              }
              %cast = tensor.cast %23 : tensor<1x64xf16> to tensor<?x?xf16>
              %24 = arith.muli %14, %c64 : index
              %25 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%arg6, %24)
              %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 64], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              %26 = loom.bufferize_to_memref %cast : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %26, %reinterpret_cast src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x64xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %20 : memref<1x64xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
}
