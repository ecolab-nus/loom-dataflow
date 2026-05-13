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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y1y8__d0i0_d1i0_d2i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (1) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y1y8__d0i1_d1i1_d2i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y2y4__d0i0_d1i0_d2i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            %23 = arith.ceildivui %21, %c16 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c4 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg3, %arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y2y4__d0i1_d1i1_d2i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %23 = arith.ceildivui %21, %c4 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c16 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg4, %arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y4y2__d0i0_d1i0_d2i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            %23 = arith.ceildivui %21, %c32 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c2 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg3, %arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y4y2__d0i1_d1i1_d2i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %23 = arith.ceildivui %21, %c2 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c32 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg4, %arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y8y1__d0i0_d1i0_d2i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            %23 = arith.ceildivui %21, %c64 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              scf.for %arg7 = %c0 to %22 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
                %25 = arith.ceildivui %c512, %20 : index
                %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
                %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %30 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
                %32 = loom.init_tensor %31[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %33 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %34 = loom.semaphore_take %33 : memref<?x?xf16> -> memref<?x?xf16>
                %35 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = scf.for %arg8 = %c0 to %25 step %c1 iter_args(%arg9 = %29) -> (tensor<?x?xf16>) {
                  %46 = arith.muli %24, %18 : index
                  %47 = arith.muli %arg8, %20 : index
                  %48 = loom.subview %arg0[%46, %47] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %48, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %34[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = arith.muli %arg7, %19 : index
                  %51 = loom.subview %arg1[%47, %50] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %51, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %52 = loom.bufferize_to_tensor %36[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %54 = linalg.matmul ins(%49, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  loom.semaphore_give %34 : memref<?x?xf16>
                  %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %56 = arith.addf %in, %in_0 : f16
                    linalg.yield %56 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %31 : memref<?x?xf16>
                  scf.yield %55 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %38 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %39 = loom.semaphore_take %38 : memref<?x?xf16> -> memref<?x?xf16>
                %40 = loom.init_tensor %39[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = linalg.copy ins(%37 : tensor<?x?xf16>) outs(%40 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                %42 = arith.muli %24, %18 : index
                %43 = arith.muli %arg7, %19 : index
                %44 = loom.subview %arg2[%42, %43] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %45, %44 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8_y8y1__d0i1_d1i1_d2i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %21 step %c1 {
              %23 = arith.ceildivui %22, %c64 : index
              scf.for %arg7 = %c0 to %23 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
                %25 = arith.ceildivui %c512, %20 : index
                %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
                %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %30 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
                %32 = loom.init_tensor %31[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %33 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %34 = loom.semaphore_take %33 : memref<?x?xf16> -> memref<?x?xf16>
                %35 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = scf.for %arg8 = %c0 to %25 step %c1 iter_args(%arg9 = %29) -> (tensor<?x?xf16>) {
                  %46 = arith.muli %arg6, %18 : index
                  %47 = arith.muli %arg8, %20 : index
                  %48 = loom.subview %arg0[%46, %47] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %48, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %34[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = arith.muli %24, %19 : index
                  %51 = loom.subview %arg1[%47, %50] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %51, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %52 = loom.bufferize_to_tensor %36[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %54 = linalg.matmul ins(%49, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  loom.semaphore_give %34 : memref<?x?xf16>
                  %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %56 = arith.addf %in, %in_0 : f16
                    linalg.yield %56 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %31 : memref<?x?xf16>
                  scf.yield %55 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %38 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %39 = loom.semaphore_take %38 : memref<?x?xf16> -> memref<?x?xf16>
                %40 = loom.init_tensor %39[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = linalg.copy ins(%37 : tensor<?x?xf16>) outs(%40 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                %42 = arith.muli %arg6, %18 : index
                %43 = arith.muli %24, %19 : index
                %44 = loom.subview %arg2[%42, %43] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %45, %44 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x1x8_y8__d0i0_d1i0_d2i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x1x8_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (1) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x2x4_y8__d0i0_d1i0_d2i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %23 = arith.ceildivui %21, %c16 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c4 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg3, %arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x2x4_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c4 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c16 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg4, %arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x4x2_y8__d0i0_d1i0_d2i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %23 = arith.ceildivui %21, %c32 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c2 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg3, %arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x4x2_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c2 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c32 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg4, %arg5, %arg7)
                %27 = arith.ceildivui %c512, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.init_tensor %33[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %35 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %48 = arith.muli %25, %18 : index
                  %49 = arith.muli %arg8, %20 : index
                  %50 = loom.subview %arg0[%48, %49] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %50, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %36[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = arith.muli %26, %19 : index
                  %53 = loom.subview %arg1[%49, %52] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %53, %38 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %54 = loom.bufferize_to_tensor %38[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %55 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %56 = linalg.matmul ins(%51, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %38 : memref<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %56 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %58 = arith.addf %in, %in_0 : f16
                    linalg.yield %58 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %57 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %40 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
                %42 = loom.init_tensor %41[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %43 = linalg.copy ins(%39 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %44 = arith.muli %25, %18 : index
                %45 = arith.muli %26, %19 : index
                %46 = loom.subview %arg2[%44, %45] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %47 = loom.bufferize_to_memref %43 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %47, %46 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %41 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8x1_y8__d0i0_d1i0_d2i1__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            %23 = arith.ceildivui %21, %c64 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              scf.for %arg7 = %c0 to %22 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
                %25 = arith.ceildivui %c512, %20 : index
                %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
                %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %30 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
                %32 = loom.init_tensor %31[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %33 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %34 = loom.semaphore_take %33 : memref<?x?xf16> -> memref<?x?xf16>
                %35 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = scf.for %arg8 = %c0 to %25 step %c1 iter_args(%arg9 = %29) -> (tensor<?x?xf16>) {
                  %46 = arith.muli %24, %18 : index
                  %47 = arith.muli %arg8, %20 : index
                  %48 = loom.subview %arg0[%46, %47] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %48, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %34[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = arith.muli %arg7, %19 : index
                  %51 = loom.subview %arg1[%47, %50] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %51, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %52 = loom.bufferize_to_tensor %36[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %54 = linalg.matmul ins(%49, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  loom.semaphore_give %34 : memref<?x?xf16>
                  %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %56 = arith.addf %in, %in_0 : f16
                    linalg.yield %56 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %31 : memref<?x?xf16>
                  scf.yield %55 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %38 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %39 = loom.semaphore_take %38 : memref<?x?xf16> -> memref<?x?xf16>
                %40 = loom.init_tensor %39[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = linalg.copy ins(%37 : tensor<?x?xf16>) outs(%40 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                %42 = arith.muli %24, %18 : index
                %43 = arith.muli %arg7, %19 : index
                %44 = loom.subview %arg2[%42, %43] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %45, %44 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @matmul__x8x1_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c4096 = arith.constant 4096 : index
      %c512 = arith.constant 512 : index
      %18 = loom.sym @tile_m {upper_bound = 4096 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 4096 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 512 : index} : index
      %21 = arith.ceildivui %c4096, %18 : index
      %22 = arith.ceildivui %c4096, %19 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %21 step %c1 {
              %23 = arith.ceildivui %22, %c64 : index
              scf.for %arg7 = %c0 to %23 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
                %25 = arith.ceildivui %c512, %20 : index
                %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
                %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %30 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
                %32 = loom.init_tensor %31[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %33 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %34 = loom.semaphore_take %33 : memref<?x?xf16> -> memref<?x?xf16>
                %35 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = scf.for %arg8 = %c0 to %25 step %c1 iter_args(%arg9 = %29) -> (tensor<?x?xf16>) {
                  %46 = arith.muli %arg6, %18 : index
                  %47 = arith.muli %arg8, %20 : index
                  %48 = loom.subview %arg0[%46, %47] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
                  loom.copy %48, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %34[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = arith.muli %24, %19 : index
                  %51 = loom.subview %arg1[%47, %50] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                  loom.copy %51, %36 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
                  %52 = loom.bufferize_to_tensor %36[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %53 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  %54 = linalg.matmul ins(%49, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%53 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %36 : memref<?x?xf16>
                  loom.semaphore_give %34 : memref<?x?xf16>
                  %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg9, %54 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) {
                  ^bb0(%in: f16, %in_0: f16, %out: f16):
                    %56 = arith.addf %in, %in_0 : f16
                    linalg.yield %56 : f16
                  } -> tensor<?x?xf16>
                  loom.semaphore_give %31 : memref<?x?xf16>
                  scf.yield %55 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %38 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %39 = loom.semaphore_take %38 : memref<?x?xf16> -> memref<?x?xf16>
                %40 = loom.init_tensor %39[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %41 = linalg.copy ins(%37 : tensor<?x?xf16>) outs(%40 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                %42 = arith.muli %arg6, %18 : index
                %43 = arith.muli %24, %19 : index
                %44 = loom.subview %arg2[%42, %43] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                %45 = loom.bufferize_to_memref %41 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %45, %44 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
                loom.semaphore_give %39 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
