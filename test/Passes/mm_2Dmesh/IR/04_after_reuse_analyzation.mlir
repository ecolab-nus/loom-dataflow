module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  %0 = adl.memory.bank "DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dram_channel", 8
  %2 = adl.memory.array "DRAM", [%1] of %0
  %3 = adl.resource "L1_torus::h"
  %4 = adl.resource "L1_torus::v"
  %5 = adl.memory.bank "bank", {bsize = 16 : i64, nblk = 5856 : i64}
  %6 = adl.spatial_dim "nbank", 16
  %7 = adl.memory.array "L1", [%6] of %5
  %8 = adl.processor.compute @matrix_lane, [(%7, %7)]
  %9 = adl.processor.compute @vector_lane, [(%7, %7)]
  %10 = adl.arch.compose "core", arch[%8, %9], mem[%7]
  %11 = adl.spatial_dim "x", 8
  %12 = adl.spatial_dim "y", 8
  %13 = adl.arch.scale "mesh", [%11, %12] of %10
  %14 = adl.processor.dmover @dram_l1_mover, [(%2, %7), (%7, %2)], with [%3, %4]
  %15 = adl.processor.dmover @dram_l1_bcst_v, [(%2, %7), (%7, %2)], with [%4]
  %16 = adl.processor.dmover @dram_l1_bcst_h, [(%2, %7), (%7, %2)], with [%3]
  %17 = adl.arch.compose "system", arch[%13, %14, %15, %16], mem[%2]
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %22 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %30 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %29) -> (tensor<?x?xf16>) {
                %35 = arith.muli %21, %18 : index
                %36 = arith.muli %arg7, %20 : index
                %37 = loom.subview %arg0[%35, %36] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %37, %25 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %25[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = arith.muli %arg6, %19 : index
                %40 = loom.subview %arg1[%36, %39] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %40, %23 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %23[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = linalg.matmul ins(%38, %41 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %23 : memref<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                affine.yield %42 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %31 = arith.muli %21, %18 : index
              %32 = arith.muli %arg6, %19 : index
              %33 = loom.subview %arg2[%31, %32] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %34 = loom.bufferize_to_memref %30 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %34, %33 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i0__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %22 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %30 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %29) -> (tensor<?x?xf16>) {
                %35 = arith.muli %21, %18 : index
                %36 = arith.muli %arg7, %20 : index
                %37 = loom.subview %arg0[%35, %36] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %37, %25 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %25[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = arith.muli %arg5, %19 : index
                %40 = loom.subview %arg1[%36, %39] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %40, %23 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %23[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = linalg.matmul ins(%38, %41 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %23 : memref<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                affine.yield %42 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %31 = arith.muli %21, %18 : index
              %32 = arith.muli %arg5, %19 : index
              %33 = loom.subview %arg2[%31, %32] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %34 = loom.bufferize_to_memref %30 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %34, %33 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %22 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %30 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %29) -> (tensor<?x?xf16>) {
                %35 = arith.muli %21, %18 : index
                %36 = arith.muli %arg7, %20 : index
                %37 = loom.subview %arg0[%35, %36] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %37, %25 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %25[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = arith.muli %arg6, %19 : index
                %40 = loom.subview %arg1[%36, %39] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %40, %23 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %23[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = linalg.matmul ins(%38, %41 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %23 : memref<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                affine.yield %42 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %31 = arith.muli %21, %18 : index
              %32 = arith.muli %arg6, %19 : index
              %33 = loom.subview %arg2[%31, %32] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %34 = loom.bufferize_to_memref %30 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %34, %33 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i0__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s1)>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 64))>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %22 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %30 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %29) -> (tensor<?x?xf16>) {
                %35 = arith.muli %21, %18 : index
                %36 = arith.muli %arg7, %20 : index
                %37 = loom.subview %arg0[%35, %36] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %37, %25 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %25[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = arith.muli %arg5, %19 : index
                %40 = loom.subview %arg1[%36, %39] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %40, %23 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %23[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = linalg.matmul ins(%38, %41 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %23 : memref<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                affine.yield %42 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %31 = arith.muli %21, %18 : index
              %32 = arith.muli %arg5, %19 : index
              %33 = loom.subview %arg2[%31, %32] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %34 = loom.bufferize_to_memref %30 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %34, %33 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %22 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %23 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.init_tensor %28[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %31 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %30) -> (tensor<?x?xf16>) {
                %36 = arith.muli %21, %18 : index
                %37 = arith.muli %arg7, %20 : index
                %38 = loom.subview %arg0[%36, %37] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %38, %26 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %39 = loom.bufferize_to_tensor %26[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = arith.muli %22, %19 : index
                %41 = loom.subview %arg1[%37, %40] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %41, %24 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %42 = loom.bufferize_to_tensor %24[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.matmul ins(%39, %42 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %24 : memref<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                affine.yield %43 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %32 = arith.muli %21, %18 : index
              %33 = arith.muli %22, %19 : index
              %34 = loom.subview %arg2[%32, %33] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %35 = loom.bufferize_to_memref %31 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %35, %34 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %28 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i0_d1i1__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %22 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %23 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.init_tensor %28[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %31 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %30) -> (tensor<?x?xf16>) {
                %36 = arith.muli %21, %18 : index
                %37 = arith.muli %arg7, %20 : index
                %38 = loom.subview %arg0[%36, %37] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %38, %26 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %39 = loom.bufferize_to_tensor %26[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = arith.muli %22, %19 : index
                %41 = loom.subview %arg1[%37, %40] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %41, %24 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %42 = loom.bufferize_to_tensor %24[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.matmul ins(%39, %42 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %24 : memref<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                affine.yield %43 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %32 = arith.muli %21, %18 : index
              %33 = arith.muli %22, %19 : index
              %34 = loom.subview %arg2[%32, %33] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %35 = loom.bufferize_to_memref %31 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %35, %34 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %28 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %22 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %23 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.init_tensor %28[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %31 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %30) -> (tensor<?x?xf16>) {
                %36 = arith.muli %21, %18 : index
                %37 = arith.muli %arg7, %20 : index
                %38 = loom.subview %arg0[%36, %37] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %38, %26 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %39 = loom.bufferize_to_tensor %26[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = arith.muli %22, %19 : index
                %41 = loom.subview %arg1[%37, %40] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %41, %24 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %42 = loom.bufferize_to_tensor %24[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.matmul ins(%39, %42 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %24 : memref<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                affine.yield %43 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %32 = arith.muli %21, %18 : index
              %33 = arith.muli %22, %19 : index
              %34 = loom.subview %arg2[%32, %33] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %35 = loom.bufferize_to_memref %31 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %35, %34 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %28 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i0_d0i1__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 8))>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s0 * 8))>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
              %22 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg5)
              %23 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
              %25 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %26 = loom.semaphore_take %25 : memref<?x?xf16> -> memref<?x?xf16>
              %27 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
              %29 = loom.init_tensor %28[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %31 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %30) -> (tensor<?x?xf16>) {
                %36 = arith.muli %21, %18 : index
                %37 = arith.muli %arg7, %20 : index
                %38 = loom.subview %arg0[%36, %37] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %38, %26 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %39 = loom.bufferize_to_tensor %26[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = arith.muli %22, %19 : index
                %41 = loom.subview %arg1[%37, %40] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %41, %24 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %42 = loom.bufferize_to_tensor %24[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.matmul ins(%39, %42 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %24 : memref<?x?xf16>
                loom.semaphore_give %26 : memref<?x?xf16>
                affine.yield %43 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %32 = arith.muli %21, %18 : index
              %33 = arith.muli %22, %19 : index
              %34 = loom.subview %arg2[%32, %33] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %35 = loom.bufferize_to_memref %31 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %35, %34 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %28 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %22 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %30 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %29) -> (tensor<?x?xf16>) {
                %35 = arith.muli %arg5, %18 : index
                %36 = arith.muli %arg7, %20 : index
                %37 = loom.subview %arg0[%35, %36] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %37, %25 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %25[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = arith.muli %21, %19 : index
                %40 = loom.subview %arg1[%36, %39] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %40, %23 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %23[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = linalg.matmul ins(%38, %41 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %23 : memref<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                affine.yield %42 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %31 = arith.muli %arg5, %18 : index
              %32 = arith.muli %21, %19 : index
              %33 = loom.subview %arg2[%31, %32] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %34 = loom.bufferize_to_memref %30 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %34, %33 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d0i1_d1i1__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %22 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %30 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %29) -> (tensor<?x?xf16>) {
                %35 = arith.muli %arg6, %18 : index
                %36 = arith.muli %arg7, %20 : index
                %37 = loom.subview %arg0[%35, %36] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %37, %25 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %25[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = arith.muli %21, %19 : index
                %40 = loom.subview %arg1[%36, %39] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %40, %23 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %23[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = linalg.matmul ins(%38, %41 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %23 : memref<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                affine.yield %42 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %31 = arith.muli %arg6, %18 : index
              %32 = arith.muli %21, %19 : index
              %33 = loom.subview %arg2[%31, %32] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %34 = loom.bufferize_to_memref %30 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %34, %33 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
              %22 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %30 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %29) -> (tensor<?x?xf16>) {
                %35 = arith.muli %arg5, %18 : index
                %36 = arith.muli %arg7, %20 : index
                %37 = loom.subview %arg0[%35, %36] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %37, %25 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %25[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = arith.muli %21, %19 : index
                %40 = loom.subview %arg1[%36, %39] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %40, %23 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %23[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = linalg.matmul ins(%38, %41 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %23 : memref<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                affine.yield %42 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %31 = arith.muli %arg5, %18 : index
              %32 = arith.muli %21, %19 : index
              %33 = loom.subview %arg2[%31, %32] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %34 = loom.bufferize_to_memref %30 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %34, %33 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
  module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
    func.func @_matmul__d1i1_d0i1__f10(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.for %arg5 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv (s1 * 64))>()[%18, %19] {
            affine.for %arg6 = 0 to affine_map<()[s0, s1] -> (4096 ceildiv s0)>()[%18, %19] {
              %21 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg5)
              %22 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
              %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
              %24 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
              %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
              %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
              %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
              %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
              %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %30 = affine.for %arg7 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%20] iter_args(%arg8 = %29) -> (tensor<?x?xf16>) {
                %35 = arith.muli %arg6, %18 : index
                %36 = arith.muli %arg7, %20 : index
                %37 = loom.subview %arg0[%35, %36] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                loom.copy %37, %25 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                %38 = loom.bufferize_to_tensor %25[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                %39 = arith.muli %21, %19 : index
                %40 = loom.subview %arg1[%36, %39] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = true] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %40, %23 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %41 = loom.bufferize_to_tensor %23[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %42 = linalg.matmul ins(%38, %41 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %23 : memref<?x?xf16>
                loom.semaphore_give %25 : memref<?x?xf16>
                affine.yield %42 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %31 = arith.muli %arg6, %18 : index
              %32 = arith.muli %21, %19 : index
              %33 = loom.subview %arg2[%31, %32] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %34 = loom.bufferize_to_memref %30 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %34, %33 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %27 : memref<?x?xf16>
            } {loom.iter_type = #loom.iter_type<temporal>}
          } {loom.iter_type = #loom.iter_type<temporal>}
        } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @x}
      } {loom.iter_type = #loom.iter_type<spatial>, loom.mapped_to = @y}
      return
    }
  }
}
