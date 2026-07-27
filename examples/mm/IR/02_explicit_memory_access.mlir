module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @matmul(%arg0: memref<2048x256xf16>, %arg1: memref<256x256xf16>, %arg2: memref<2048x256xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %0 = loom.sym @tile_m {upper_bound = 2048 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 256 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 256 : index} : index
    %3 = arith.ceildivui %c2048, %0 : index
    %4 = arith.ceildivui %c256, %1 : index
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
        %23 = arith.muli %arg3, %0 : index
        %24 = arith.muli %arg5, %2 : index
        %25 = loom.subview %arg0[%23, %24] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %25, %11 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %26 = loom.bufferize_to_tensor %11[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %27 = arith.muli %arg4, %1 : index
        %28 = loom.subview %arg1[%24, %27] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %28, %13 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %29 = loom.bufferize_to_tensor %13[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %30 = linalg.matmul ins(%26, %29 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg6 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %13 : memref<?x?xf16>
        loom.semaphore_give %11 : memref<?x?xf16>
        scf.yield %30 : tensor<?x?xf16>
      }
      %15 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %16 = loom.semaphore_take %15 : memref<?x?xf16> -> memref<?x?xf16>
      %17 = loom.init_tensor %16[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %18 = linalg.copy ins(%14 : tensor<?x?xf16>) outs(%17 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %7 : memref<?x?xf16>
      %19 = arith.muli %arg3, %0 : index
      %20 = arith.muli %arg4, %1 : index
      %21 = loom.subview %arg2[%19, %20] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      %22 = loom.bufferize_to_memref %18 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %22, %21 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      loom.semaphore_give %16 : memref<?x?xf16>
    }
    return
  }
}
