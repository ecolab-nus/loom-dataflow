module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @split_k_matmul(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 256 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 4096 : index} : index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (256 ceildiv symbol(%0), 256 ceildiv symbol(%1), 4096 ceildiv symbol(%2)) {
      %3 = arith.muli %arg3, %0 : index
      %4 = arith.muli %arg4, %1 : index
      %5 = arith.muli %arg5, %2 : index
      %6 = loom.alloc [%0, %2] on @L1 : memref<?x?xf32>
      %7 = loom.semaphore_take %6 : memref<?x?xf32> -> memref<?x?xf32>
      %8 = loom.subview %arg1[%3, %5] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      loom.copy %8, %7 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32>
      %9 = loom.bufferize_to_tensor %7[%0, %2] : memref<?x?xf32> -> tensor<?x?xf32>
      %10 = loom.alloc [%2, %1] on @L1 : memref<?x?xf32>
      %11 = loom.semaphore_take %10 : memref<?x?xf32> -> memref<?x?xf32>
      %12 = loom.subview %arg2[%5, %4] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      loom.copy %12, %11 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32>
      %13 = loom.bufferize_to_tensor %11[%2, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %14 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
      %15 = loom.semaphore_take %14 : memref<?x?xf32> -> memref<?x?xf32>
      %16 = loom.init_tensor %15[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %17 = linalg.fill ins(%cst : f32) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %18 = linalg.matmul ins(%9, %13 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
      loom.semaphore_give %11 : memref<?x?xf32>
      loom.semaphore_give %7 : memref<?x?xf32>
      %19 = arith.cmpi eq, %5, %c0 : index
      scf.if %19 {
        %20 = loom.reduce_sum %18(UB : [%c0, %c0], LB : [%c0, %c0]) : tensor<?x?xf32> -> tensor<?x?xf32>
        %21 = loom.subview %arg0[%3, %4] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        %22 = loom.bufferize_to_memref %20 : tensor<?x?xf32> -> memref<?x?xf32>
        loom.copy %22, %21 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        loom.semaphore_give %15 : memref<?x?xf32>
      }
    }
    return
  }
}
