module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.memory.bank "mem_bank", {bsize = 16 : i64, nblk = 5464 : i64}
  %4 = adl.spatial_dim "dim_nbank", 16
  %5 = adl.memory.array "mem_L1", [%4] of %3
  %6 = adl.resource.exclusive "res_matrix_lane"
  %7 = adl.resource.exclusive "res_vector_lane"
  %8 = adl.processor.compute @proc_matrix_lane, [(%5, %5)], with [%6]
  %9 = adl.processor.compute @proc_vector_lane, [(%5, %5)], with [%7]
  %10 = adl.arch.compose "arch_core", arch[%8, %9], mem[%5]
  %11 = adl.spatial_dim "dim_x", 8
  %12 = adl.spatial_dim "dim_y", 8
  %13 = adl.memory.array "mem_array_L1", [%11, %12] of %5
  %14 = adl.arch.scale "arch_mesh", [%11, %12] of %10, mem_region %13
  %15 = adl.processor.dmover @proc_dram_l1_noc0, [(%2, %13)]
  %16 = adl.processor.dmover @proc_dram_l1_noc1, [(%13, %2), (%13, %13)]
  %17 = adl.arch.compose "arch_system", arch[%14, %15, %16], mem[%2]
  func.func @matmul(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
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
    %23 = loom.sym @ld_00 : index
    %24 = loom.sym @ld_01 : index
    %25 = loom.sym @ld_10 : index
    %26 = loom.sym @ld_11 : index
    %map, %lds:2 = loom.mapping_matrix @arch_mesh [[%23, %24], [%25, %26]] : !loom.spatial_map<2 x 2>
    loom.spatial_mapping (%arg3, %arg4) to (%21, %22) using %map : !loom.spatial_map<2 x 2> ld [%lds#0, %lds#1] waves(%arg5, %arg6) {
      %27 = arith.ceildivui %c256, %20 : index
      %28 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
      %29 = loom.semaphore_take %28 : memref<?x?xf16> -> memref<?x?xf16>
      %30 = loom.init_tensor %29[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
      %31 = linalg.fill ins(%cst : f16) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %32 = loom.alloc [%18, %20] on @L1 : memref<?x?xf16>
      %33 = loom.semaphore_take %32 : memref<?x?xf16> -> memref<?x?xf16>
      %34 = loom.alloc [%20, %19] on @L1 : memref<?x?xf16>
      %35 = loom.semaphore_take %34 : memref<?x?xf16> -> memref<?x?xf16>
      %36 = scf.for %arg7 = %c0 to %27 step %c1 iter_args(%arg8 = %31) -> (tensor<?x?xf16>) {
        %45 = arith.muli %arg3, %18 : index
        %46 = arith.muli %arg7, %20 : index
        %47 = loom.subview %arg0[%45, %46] [%18, %20] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %47, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, %lds#1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %48 = loom.bufferize_to_tensor %33[%18, %20] : memref<?x?xf16> -> tensor<?x?xf16>
        %49 = arith.muli %arg4, %19 : index
        %50 = loom.subview %arg1[%46, %49] [%20, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %50, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [%lds#0, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %51 = loom.bufferize_to_tensor %35[%20, %19] : memref<?x?xf16> -> tensor<?x?xf16>
        %52 = linalg.matmul ins(%48, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %35 : memref<?x?xf16>
        loom.semaphore_give %33 : memref<?x?xf16>
        scf.yield %52 : tensor<?x?xf16>
      }
      %37 = loom.alloc [%18, %19] on @L1 : memref<?x?xf16>
      %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
      %39 = loom.init_tensor %38[%18, %19] : memref<?x?xf16> -> tensor<?x?xf16>
      %40 = linalg.copy ins(%36 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %29 : memref<?x?xf16>
      %41 = arith.muli %arg3, %18 : index
      %42 = arith.muli %arg4, %19 : index
      %43 = loom.subview %arg2[%41, %42] [%18, %19] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      %44 = loom.bufferize_to_memref %40 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %44, %43 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      loom.semaphore_give %38 : memref<?x?xf16>
    }
    return
  }
}
