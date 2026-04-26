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
    func.func @_matmul__x8_y8__d0i0_d1i1__f01__dim_y_level0_bc8_dim_x_level0_bc8_n__tile_k64__tile_m1__tile_n1024(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          scf.for %arg5 = %c0 to %c512 step %c1 {
            %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
            %21 = loom.alloc [1, 1024] on @L1 : memref<1x1024xf16>
            %22 = loom.semaphore_take %21 : memref<1x1024xf16> -> memref<1x1024xf16>
            %23 = loom.init_tensor %22[1, 1024] : memref<1x1024xf16> -> tensor<1x1024xf16>
            %24 = loom.semaphore_take %21 : memref<1x1024xf16> -> memref<1x1024xf16>
            %25 = loom.init_tensor %24[1, 1024] : memref<1x1024xf16> -> tensor<1x1024xf16>
            %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
            %27 = loom.alloc [1, 1024] on @L1 : memref<1x1024xf16>
            %28 = loom.semaphore_take %27 : memref<1x1024xf16> -> memref<1x1024xf16>
            %29 = loom.init_tensor %28[1, 1024] : memref<1x1024xf16> -> tensor<1x1024xf16>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %31 = loom.semaphore_take %30 : memref<1x64xf16> -> memref<1x64xf16>
            %32 = loom.init_tensor %31[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
            %33 = loom.semaphore_take %30 : memref<1x64xf16> -> memref<1x64xf16>
            %34 = loom.alloc [64, 1024] on @L1 : memref<64x1024xf16>
            %35 = loom.semaphore_take %34 : memref<64x1024xf16> -> memref<64x1024xf16>
            %36 = loom.init_tensor %35[64, 1024] : memref<64x1024xf16> -> tensor<64x1024xf16>
            %37 = loom.semaphore_take %34 : memref<64x1024xf16> -> memref<64x1024xf16>
            %38 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %26) -> (tensor<1x1024xf16>) {
              %43 = arith.muli %arg6, %c64 : index
              %44 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%20, %43)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [1, 64], strides: [256, 1] : memref<4096x256xf16> to memref<1x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%arg3, %c0], LR : [%arg3, %c7]) : memref<1x64xf16, strided<[256, 1], offset: ?>> to memref<1x64xf16>
              %45 = loom.bufferize_to_tensor %33[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %46 = loom.sync ins(%45 : tensor<1x64xf16>) outs(%32 : tensor<1x64xf16>) -> tensor<1x64xf16>
              loom.semaphore_give %33 : memref<1x64xf16>
              %47 = arith.muli %arg4, %c1024 : index
              %48 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%43, %47)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%48], sizes: [64, 1024], strides: [4096, 1] : memref<256x4096xf16> to memref<64x1024xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<64x1024xf16, strided<[4096, 1], offset: ?>> to memref<64x1024xf16>
              %49 = loom.bufferize_to_tensor %37[64, 1024] : memref<64x1024xf16> -> tensor<64x1024xf16>
              %50 = loom.sync ins(%49 : tensor<64x1024xf16>) outs(%36 : tensor<64x1024xf16>) -> tensor<64x1024xf16>
              loom.semaphore_give %37 : memref<64x1024xf16>
              %51 = linalg.fill ins(%cst : f16) outs(%29 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
              %52 = linalg.matmul ins(%46, %50 : tensor<1x64xf16>, tensor<64x1024xf16>) outs(%51 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
              loom.semaphore_give %35 : memref<64x1024xf16>
              loom.semaphore_give %31 : memref<1x64xf16>
              %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %52 : tensor<1x1024xf16>, tensor<1x1024xf16>) outs(%arg7 : tensor<1x1024xf16>) {
              ^bb0(%in: f16, %in_2: f16, %out: f16):
                %54 = arith.addf %in, %in_2 : f16
                linalg.yield %54 : f16
              } -> tensor<1x1024xf16>
              loom.semaphore_give %28 : memref<1x1024xf16>
              scf.yield %53 : tensor<1x1024xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            %39 = arith.muli %arg4, %c1024 : index
            %40 = loom.sync ins(%38 : tensor<1x1024xf16>) outs(%23 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
            loom.semaphore_give %24 : memref<1x1024xf16>
            %41 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %39)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 1024], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x1024xf16, strided<[4096, 1], offset: ?>>
            %42 = loom.bufferize_to_memref %40 : tensor<1x1024xf16> -> memref<1x1024xf16>
            loom.copy %42, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg3, %arg4], LR : [%arg3, %arg4]) : memref<1x1024xf16> to memref<1x1024xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %22 : memref<1x1024xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @_matmul__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_dim_y_level0_bc8_n__tile_k64__tile_m1__tile_n1024(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c512 = arith.constant 512 : index
      %c7 = arith.constant 7 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          scf.for %arg5 = %c0 to %c512 step %c1 {
            %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg5)
            %21 = loom.alloc [1, 1024] on @L1 : memref<1x1024xf16>
            %22 = loom.semaphore_take %21 : memref<1x1024xf16> -> memref<1x1024xf16>
            %23 = loom.init_tensor %22[1, 1024] : memref<1x1024xf16> -> tensor<1x1024xf16>
            %24 = loom.semaphore_take %21 : memref<1x1024xf16> -> memref<1x1024xf16>
            %25 = loom.init_tensor %24[1, 1024] : memref<1x1024xf16> -> tensor<1x1024xf16>
            %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
            %27 = loom.alloc [1, 1024] on @L1 : memref<1x1024xf16>
            %28 = loom.semaphore_take %27 : memref<1x1024xf16> -> memref<1x1024xf16>
            %29 = loom.init_tensor %28[1, 1024] : memref<1x1024xf16> -> tensor<1x1024xf16>
            %30 = loom.alloc [1, 64] on @L1 : memref<1x64xf16>
            %31 = loom.semaphore_take %30 : memref<1x64xf16> -> memref<1x64xf16>
            %32 = loom.init_tensor %31[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
            %33 = loom.semaphore_take %30 : memref<1x64xf16> -> memref<1x64xf16>
            %34 = loom.alloc [64, 1024] on @L1 : memref<64x1024xf16>
            %35 = loom.semaphore_take %34 : memref<64x1024xf16> -> memref<64x1024xf16>
            %36 = loom.init_tensor %35[64, 1024] : memref<64x1024xf16> -> tensor<64x1024xf16>
            %37 = loom.semaphore_take %34 : memref<64x1024xf16> -> memref<64x1024xf16>
            %38 = scf.for %arg6 = %c0 to %c4 step %c1 iter_args(%arg7 = %26) -> (tensor<1x1024xf16>) {
              %43 = arith.muli %arg6, %c64 : index
              %44 = affine.apply affine_map<(d0, d1) -> (d0 * 256 + d1)>(%20, %43)
              %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%44], sizes: [1, 64], strides: [256, 1] : memref<4096x256xf16> to memref<1x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_0, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg3], LR : [%c7, %arg3]) : memref<1x64xf16, strided<[256, 1], offset: ?>> to memref<1x64xf16>
              %45 = loom.bufferize_to_tensor %33[1, 64] : memref<1x64xf16> -> tensor<1x64xf16>
              %46 = loom.sync ins(%45 : tensor<1x64xf16>) outs(%32 : tensor<1x64xf16>) -> tensor<1x64xf16>
              loom.semaphore_give %33 : memref<1x64xf16>
              %47 = arith.muli %arg4, %c1024 : index
              %48 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%43, %47)
              %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [%48], sizes: [64, 1024], strides: [4096, 1] : memref<256x4096xf16> to memref<64x1024xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_1, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<64x1024xf16, strided<[4096, 1], offset: ?>> to memref<64x1024xf16>
              %49 = loom.bufferize_to_tensor %37[64, 1024] : memref<64x1024xf16> -> tensor<64x1024xf16>
              %50 = loom.sync ins(%49 : tensor<64x1024xf16>) outs(%36 : tensor<64x1024xf16>) -> tensor<64x1024xf16>
              loom.semaphore_give %37 : memref<64x1024xf16>
              %51 = linalg.fill ins(%cst : f16) outs(%29 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
              %52 = linalg.matmul ins(%46, %50 : tensor<1x64xf16>, tensor<64x1024xf16>) outs(%51 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
              loom.semaphore_give %35 : memref<64x1024xf16>
              loom.semaphore_give %31 : memref<1x64xf16>
              %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %52 : tensor<1x1024xf16>, tensor<1x1024xf16>) outs(%arg7 : tensor<1x1024xf16>) {
              ^bb0(%in: f16, %in_2: f16, %out: f16):
                %54 = arith.addf %in, %in_2 : f16
                linalg.yield %54 : f16
              } -> tensor<1x1024xf16>
              loom.semaphore_give %28 : memref<1x1024xf16>
              scf.yield %53 : tensor<1x1024xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            %39 = arith.muli %arg4, %c1024 : index
            %40 = loom.sync ins(%38 : tensor<1x1024xf16>) outs(%23 : tensor<1x1024xf16>) -> tensor<1x1024xf16>
            loom.semaphore_give %24 : memref<1x1024xf16>
            %41 = affine.apply affine_map<(d0, d1) -> (d0 * 4096 + d1)>(%20, %39)
            %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%41], sizes: [1, 1024], strides: [4096, 1] : memref<4096x4096xf16> to memref<1x1024xf16, strided<[4096, 1], offset: ?>>
            %42 = loom.bufferize_to_memref %40 : tensor<1x1024xf16> -> memref<1x1024xf16>
            loom.copy %42, %reinterpret_cast src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg3], LR : [%arg4, %arg3]) : memref<1x1024xf16> to memref<1x1024xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %22 : memref<1x1024xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
}
