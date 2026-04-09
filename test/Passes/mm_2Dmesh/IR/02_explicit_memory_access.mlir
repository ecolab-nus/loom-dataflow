module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @_matmul(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %0 = loom.sym @tile_m {upper_bound = 4096 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 4096 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 256 : index} : index
    %3 = arith.ceildivui %c4096, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = arith.ceildivui %c256, %2 : index
      %6 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %7 = loom.semaphore_take %6 : memref<?x?xf16> -> memref<?x?xf16>
      %8 = loom.init_tensor %7[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %10 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %11 = loom.semaphore_take %10 : memref<?x?xf16> -> memref<?x?xf16>
      %12 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %13 = loom.semaphore_take %12 : memref<?x?xf16> -> memref<?x?xf16>
      %14 = scf.for %arg5 = %c0 to %5 step %c1 iter_args(%arg6 = %9) -> (tensor<?x?xf16>) {
        %19 = arith.muli %arg3, %0 : index
        %20 = arith.muli %arg5, %2 : index
        %21 = loom.subview %arg0[%19, %20] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %21, %11 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %22 = loom.bufferize_to_tensor %11[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %23 = arith.muli %arg4, %1 : index
        %24 = loom.subview %arg1[%20, %23] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        loom.copy %24, %13 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
        %25 = loom.bufferize_to_tensor %13[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %26 = linalg.matmul ins(%22, %25 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg6 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %13 : memref<?x?xf16>
        loom.semaphore_give %11 : memref<?x?xf16>
        scf.yield %26 : tensor<?x?xf16>
      }
      %15 = arith.muli %arg3, %0 : index
      %16 = arith.muli %arg4, %1 : index
      %17 = loom.subview %arg2[%15, %16] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %18 = loom.bufferize_to_memref %14 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %18, %17 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %7 : memref<?x?xf16>
    }
    return
  }
}
