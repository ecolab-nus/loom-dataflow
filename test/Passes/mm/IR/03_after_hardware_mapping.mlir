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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i0_d1i1__f01(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %23 = arith.ceildivui %c4096, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          %25 = arith.ceildivui %23, %c8 : index
          scf.for %arg5 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %29 = arith.ceildivui %c256, %22 : index
              %30 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
              %32 = loom.init_tensor %31[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %33 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
              %34 = loom.init_tensor %33[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %35 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %36 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %37 = loom.semaphore_take %36 : memref<?x?xf16> -> memref<?x?xf16>
              %38 = loom.init_tensor %37[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %39 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %40 = loom.semaphore_take %39 : memref<?x?xf16> -> memref<?x?xf16>
              %41 = loom.init_tensor %40[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
              %42 = loom.semaphore_take %39 : memref<?x?xf16> -> memref<?x?xf16>
              %43 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %44 = loom.semaphore_take %43 : memref<?x?xf16> -> memref<?x?xf16>
              %45 = loom.init_tensor %44[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %46 = loom.semaphore_take %43 : memref<?x?xf16> -> memref<?x?xf16>
              %47 = scf.for %arg7 = %c0 to %29 step %c1 iter_args(%arg8 = %35) -> (tensor<?x?xf16>) {
                %53 = arith.muli %27, %20 : index
                %54 = arith.muli %arg7, %22 : index
                %55 = loom.subview %arg0[%53, %54] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.copy %55, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                %56 = loom.bufferize_to_tensor %42[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %57 = loom.sync ins(%56 : tensor<?x?xf16>) outs(%41 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %42 : memref<?x?xf16>
                %58 = arith.muli %28, %21 : index
                %59 = loom.subview %arg1[%54, %58] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %59, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %60 = loom.bufferize_to_tensor %46[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %61 = loom.sync ins(%60 : tensor<?x?xf16>) outs(%45 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %46 : memref<?x?xf16>
                %62 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %63 = linalg.matmul ins(%57, %61 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %44 : memref<?x?xf16>
                loom.semaphore_give %40 : memref<?x?xf16>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %63 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_0: f16, %out: f16):
                  %65 = arith.addf %in, %in_0 : f16
                  linalg.yield %65 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %37 : memref<?x?xf16>
                scf.yield %64 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %48 = arith.muli %27, %20 : index
              %49 = arith.muli %28, %21 : index
              %50 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%32 : tensor<?x?xf16>) -> tensor<?x?xf16>
              loom.semaphore_give %33 : memref<?x?xf16>
              %51 = loom.subview %arg2[%48, %49] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %52 = loom.bufferize_to_memref %50 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %52, %51 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %31 : memref<?x?xf16>
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i1_d1i0__f01(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c256 = arith.constant 256 : index
      %20 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %21 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %22 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %23 = arith.ceildivui %c4096, %20 : index
      %24 = arith.ceildivui %c4096, %21 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          %25 = arith.ceildivui %23, %c8 : index
          scf.for %arg5 = %c0 to %25 step %c1 {
            %26 = arith.ceildivui %24, %c8 : index
            scf.for %arg6 = %c0 to %26 step %c1 {
              %27 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
              %28 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
              %29 = arith.ceildivui %c256, %22 : index
              %30 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
              %32 = loom.init_tensor %31[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %33 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
              %34 = loom.init_tensor %33[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %35 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
              %36 = loom.alloc [%20, %21] on @L1 : memref<?x?xf16>
              %37 = loom.semaphore_take %36 : memref<?x?xf16> -> memref<?x?xf16>
              %38 = loom.init_tensor %37[%20, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %39 = loom.alloc [%20, %22] on @L1 : memref<?x?xf16>
              %40 = loom.semaphore_take %39 : memref<?x?xf16> -> memref<?x?xf16>
              %41 = loom.init_tensor %40[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
              %42 = loom.semaphore_take %39 : memref<?x?xf16> -> memref<?x?xf16>
              %43 = loom.alloc [%22, %21] on @L1 : memref<?x?xf16>
              %44 = loom.semaphore_take %43 : memref<?x?xf16> -> memref<?x?xf16>
              %45 = loom.init_tensor %44[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
              %46 = loom.semaphore_take %43 : memref<?x?xf16> -> memref<?x?xf16>
              %47 = scf.for %arg7 = %c0 to %29 step %c1 iter_args(%arg8 = %35) -> (tensor<?x?xf16>) {
                %53 = arith.muli %27, %20 : index
                %54 = arith.muli %arg7, %22 : index
                %55 = loom.subview %arg0[%53, %54] [%20, %22] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.copy %55, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                %56 = loom.bufferize_to_tensor %42[%20, %22] : memref<?x?xf16> -> tensor<?x?xf16>
                %57 = loom.sync ins(%56 : tensor<?x?xf16>) outs(%41 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %42 : memref<?x?xf16>
                %58 = arith.muli %28, %21 : index
                %59 = loom.subview %arg1[%54, %58] [%22, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.copy %59, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                %60 = loom.bufferize_to_tensor %46[%22, %21] : memref<?x?xf16> -> tensor<?x?xf16>
                %61 = loom.sync ins(%60 : tensor<?x?xf16>) outs(%45 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %46 : memref<?x?xf16>
                %62 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %63 = linalg.matmul ins(%57, %61 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%62 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %44 : memref<?x?xf16>
                loom.semaphore_give %40 : memref<?x?xf16>
                %64 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %63 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) {
                ^bb0(%in: f16, %in_0: f16, %out: f16):
                  %65 = arith.addf %in, %in_0 : f16
                  linalg.yield %65 : f16
                } -> tensor<?x?xf16>
                loom.semaphore_give %37 : memref<?x?xf16>
                scf.yield %64 : tensor<?x?xf16>
              } {loom.iter_type = #loom.iter_type<sequential>}
              %48 = arith.muli %27, %20 : index
              %49 = arith.muli %28, %21 : index
              %50 = loom.sync ins(%47 : tensor<?x?xf16>) outs(%32 : tensor<?x?xf16>) -> tensor<?x?xf16>
              loom.semaphore_give %33 : memref<?x?xf16>
              %51 = loom.subview %arg2[%48, %49] [%20, %21] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              %52 = loom.bufferize_to_memref %50 : tensor<?x?xf16> -> memref<?x?xf16>
              loom.copy %52, %51 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
              loom.semaphore_give %31 : memref<?x?xf16>
            } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
