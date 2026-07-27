module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
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
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8_y1y8__d0i0_d1i0_d2i1__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (1) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = true, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8_y1y8__d0i1_d1i1_d2i0__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = true, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8_y2y4__d0i0_d1i0_d2i1__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (4) {
            %23 = arith.ceildivui %21, %c16 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c4 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg3, %arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8_y2y4__d0i1_d1i1_d2i0__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %23 = arith.ceildivui %21, %c4 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c16 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 16)>(%arg4, %arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8_y4y2__d0i0_d1i0_d2i1__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (2) {
            %23 = arith.ceildivui %21, %c32 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c2 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg3, %arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8_y4y2__d0i1_d1i1_d2i0__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %23 = arith.ceildivui %21, %c2 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c32 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 32)>(%arg4, %arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8_y8y1__d0i0_d1i0_d2i1__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            %23 = arith.ceildivui %21, %c64 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              scf.for %arg7 = %c0 to %22 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
                %25 = arith.ceildivui %c256, %20 : index
                %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
                %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %30 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
                %32 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = scf.for %arg8 = %c0 to %25 step %c1 iter_args(%arg9 = %29) -> (tensor<?x?xf16>) {
                  %43 = arith.muli %24, %18 : index
                  %44 = arith.muli %arg8, %20 : index
                  %45 = loom.subview %arg0[%43, %44] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %45, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %31[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = arith.muli %arg7, %19 : index
                  %48 = loom.subview %arg1[%44, %47] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %48, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %33[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.matmul ins(%46, %49 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  loom.semaphore_give %31 : memref<?x?xf16>
                  scf.yield %50 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %35 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.init_tensor %36[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.copy ins(%34 : tensor<?x?xf16>) outs(%37 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                %39 = arith.muli %24, %18 : index
                %40 = arith.muli %arg7, %19 : index
                %41 = loom.subview %arg2[%39, %40] [%18, %19] [1, 1], reuse : [seq = false, spat = true, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %42 = loom.bufferize_to_memref %38 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %42, %41 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %36 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8_y8y1__d0i1_d1i1_d2i0__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %21 step %c1 {
              %23 = arith.ceildivui %22, %c64 : index
              scf.for %arg7 = %c0 to %23 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
                %25 = arith.ceildivui %c256, %20 : index
                %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
                %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %30 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
                %32 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = scf.for %arg8 = %c0 to %25 step %c1 iter_args(%arg9 = %29) -> (tensor<?x?xf16>) {
                  %43 = arith.muli %arg6, %18 : index
                  %44 = arith.muli %arg8, %20 : index
                  %45 = loom.subview %arg0[%43, %44] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %45, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %31[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = arith.muli %24, %19 : index
                  %48 = loom.subview %arg1[%44, %47] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %48, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %33[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.matmul ins(%46, %49 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  loom.semaphore_give %31 : memref<?x?xf16>
                  scf.yield %50 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %35 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.init_tensor %36[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.copy ins(%34 : tensor<?x?xf16>) outs(%37 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                %39 = arith.muli %arg6, %18 : index
                %40 = arith.muli %24, %19 : index
                %41 = loom.subview %arg2[%39, %40] [%18, %19] [1, 1], reuse : [seq = false, spat = true, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %42 = loom.bufferize_to_memref %38 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %42, %41 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %36 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x1x8_y8__d0i0_d1i0_d2i1__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = true, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x1x8_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (1) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c8 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c8 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 8)>(%arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = true, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x2x4_y8__d0i0_d1i0_d2i1__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (4) {
            %23 = arith.ceildivui %21, %c16 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c4 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg3, %arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x2x4_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c16 = arith.constant 16 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (2) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c4 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c16 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 4)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 2 + d1 + d2 * 16)>(%arg4, %arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x4x2_y8__d0i0_d1i0_d2i1__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (4) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (2) {
            %23 = arith.ceildivui %21, %c32 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c2 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg3, %arg4, %arg6)
                %26 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x4x2_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (2) {
        affine.parallel (%arg4) = (0) to (4) {
          affine.parallel (%arg5) = (0) to (8) {
            %23 = arith.ceildivui %21, %c2 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              %24 = arith.ceildivui %22, %c32 : index
              scf.for %arg7 = %c0 to %24 step %c1 {
                %25 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg3, %arg6)
                %26 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 4 + d1 + d2 * 32)>(%arg4, %arg5, %arg7)
                %27 = arith.ceildivui %c256, %20 : index
                %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
                %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
                %36 = scf.for %arg8 = %c0 to %27 step %c1 iter_args(%arg9 = %31) -> (tensor<?x?xf16>) {
                  %45 = arith.muli %25, %18 : index
                  %46 = arith.muli %arg8, %20 : index
                  %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %49 = arith.muli %26, %19 : index
                  %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %35 : memref<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  scf.yield %52 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
                %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %29 : memref<?x?xf16>
                %41 = arith.muli %25, %18 : index
                %42 = arith.muli %26, %19 : index
                %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %38 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8x1_y8__d0i0_d1i0_d2i1__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (8) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (1) {
            %23 = arith.ceildivui %21, %c64 : index
            scf.for %arg6 = %c0 to %23 step %c1 {
              scf.for %arg7 = %c0 to %22 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg3, %arg4, %arg6)
                %25 = arith.ceildivui %c256, %20 : index
                %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
                %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %30 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
                %32 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = scf.for %arg8 = %c0 to %25 step %c1 iter_args(%arg9 = %29) -> (tensor<?x?xf16>) {
                  %43 = arith.muli %24, %18 : index
                  %44 = arith.muli %arg8, %20 : index
                  %45 = loom.subview %arg0[%43, %44] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %45, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %31[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = arith.muli %arg7, %19 : index
                  %48 = loom.subview %arg1[%44, %47] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %48, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %33[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.matmul ins(%46, %49 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  loom.semaphore_give %31 : memref<?x?xf16>
                  scf.yield %50 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %35 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.init_tensor %36[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.copy ins(%34 : tensor<?x?xf16>) outs(%37 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                %39 = arith.muli %24, %18 : index
                %40 = arith.muli %arg7, %19 : index
                %41 = loom.subview %arg2[%39, %40] [%18, %19] [1, 1], reuse : [seq = false, spat = true, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %42 = loom.bufferize_to_memref %38 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %42, %41 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %36 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
    func.func @matmul__x8x1_y8__d0i1_d1i1_d2i0__f01(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %18 = loom.sym @tile_m {upper_bound = 2048 : index} : index
      %19 = loom.sym @tile_n {upper_bound = 256 : index} : index
      %20 = loom.sym @tile_k {upper_bound = 256 : index} : index
      %21 = arith.ceildivui %c2048, %18 : index
      %22 = arith.ceildivui %c256, %19 : index
      affine.parallel (%arg3) = (0) to (1) {
        affine.parallel (%arg4) = (0) to (8) {
          affine.parallel (%arg5) = (0) to (8) {
            scf.for %arg6 = %c0 to %21 step %c1 {
              %23 = arith.ceildivui %22, %c64 : index
              scf.for %arg7 = %c0 to %23 step %c1 {
                %24 = affine.apply affine_map<(d0, d1, d2) -> (d0 * 8 + d1 + d2 * 64)>(%arg4, %arg5, %arg7)
                %25 = arith.ceildivui %c256, %20 : index
                %26 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %27 = loom.semaphore_take %26 : memref<?x?xf16> -> memref<?x?xf16>
                %28 = loom.init_tensor %27[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
                %30 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
                %31 = loom.semaphore_take %30 : memref<?x?xf16> -> memref<?x?xf16>
                %32 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
                %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
                %34 = scf.for %arg8 = %c0 to %25 step %c1 iter_args(%arg9 = %29) -> (tensor<?x?xf16>) {
                  %43 = arith.muli %arg6, %18 : index
                  %44 = arith.muli %arg8, %20 : index
                  %45 = loom.subview %arg0[%43, %44] [%18, %20] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %45, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %46 = loom.bufferize_to_tensor %31[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
                  %47 = arith.muli %24, %19 : index
                  %48 = loom.subview %arg1[%44, %47] [%20, %19] [1, 1], reuse : [seq = false, spat = true, temp = true] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %48, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
                  %49 = loom.bufferize_to_tensor %33[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                  %50 = linalg.matmul ins(%46, %49 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?xf16>) -> tensor<?x?xf16>
                  loom.semaphore_give %33 : memref<?x?xf16>
                  loom.semaphore_give %31 : memref<?x?xf16>
                  scf.yield %50 : tensor<?x?xf16>
                } {loom.iter_type = #loom.iter_type<sequential>}
                %35 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
                %36 = loom.semaphore_take %35 : memref<?x?xf16> -> memref<?x?xf16>
                %37 = loom.init_tensor %36[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
                %38 = linalg.copy ins(%34 : tensor<?x?xf16>) outs(%37 : tensor<?x?xf16>) -> tensor<?x?xf16>
                loom.semaphore_give %27 : memref<?x?xf16>
                %39 = arith.muli %arg6, %18 : index
                %40 = arith.muli %24, %19 : index
                %41 = loom.subview %arg2[%39, %40] [%18, %19] [1, 1], reuse : [seq = false, spat = true, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                %42 = loom.bufferize_to_memref %38 : tensor<?x?xf16> -> memref<?x?xf16>
                loom.copy %42, %41 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                loom.semaphore_give %36 : memref<?x?xf16>
              } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<temporal>}
            } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
