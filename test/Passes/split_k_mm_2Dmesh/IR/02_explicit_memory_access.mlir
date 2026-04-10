module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @split_k_matmul(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 256 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 4096 : index} : index
    %3 = arith.ceildivui %c256, %0 : index
    %4 = arith.ceildivui %c256, %1 : index
    %5 = arith.ceildivui %c4096, %2 : index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%3), symbol(%4), symbol(%5)) {
      %6 = arith.muli %arg3, %0 : index
      %7 = arith.muli %arg5, %2 : index
      %8 = loom.alloc [%0, %2] on @L1 : memref<?x?xf32>
      %9 = loom.semaphore_take %8 : memref<?x?xf32> -> memref<?x?xf32>
      %10 = loom.subview %arg1[%6, %7] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      loom.copy %10, %9 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
      %11 = loom.bufferize_to_tensor %9[%0, %2] : memref<?x?xf32> -> tensor<?x?xf32>
      %12 = arith.muli %arg4, %1 : index
      %13 = loom.alloc [%2, %1] on @L1 : memref<?x?xf32>
      %14 = loom.semaphore_take %13 : memref<?x?xf32> -> memref<?x?xf32>
      %15 = loom.subview %arg2[%7, %12] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      loom.copy %15, %14 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
      %16 = loom.bufferize_to_tensor %14[%2, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %17 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
      %18 = loom.semaphore_take %17 : memref<?x?xf32> -> memref<?x?xf32>
      %19 = loom.init_tensor %18[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %20 = linalg.fill ins(%cst : f32) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %21 = linalg.matmul ins(%11, %16 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%20 : tensor<?x?xf32>) -> tensor<?x?xf32>
      loom.semaphore_give %14 : memref<?x?xf32>
      loom.semaphore_give %9 : memref<?x?xf32>
      %22 = arith.cmpi eq, %arg5, %c0 : index
      %23 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
      %24 = loom.semaphore_take %23 : memref<?x?xf32> -> memref<?x?xf32>
      %25 = loom.init_tensor %24[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      scf.if %22 {
        %26 = linalg.fill ins(%cst : f32) outs(%25 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %27 = loom.reduce_sum ins(%21) outs(%26) (UB : [%c0, %c0], LB : [%c0, %c0]) : tensor<?x?xf32> -> tensor<?x?xf32>
        loom.semaphore_give %18 : memref<?x?xf32>
        %28 = loom.subview %arg0[%6, %12] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        %29 = loom.bufferize_to_memref %27 : tensor<?x?xf32> -> memref<?x?xf32>
        loom.copy %29, %28 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        loom.semaphore_give %24 : memref<?x?xf32>
      }
    }
    return
  }
}
