module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @batch_matmul(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = loom.sym @tile_b {upper_bound = 8 : index} : index
    %1 = loom.sym @tile_m {upper_bound = 4096 : index} : index
    %2 = loom.sym @tile_n {upper_bound = 4096 : index} : index
    %3 = loom.sym @tile_k {upper_bound = 512 : index} : index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (8 ceildiv symbol(%0), 4096 ceildiv symbol(%1), 4096 ceildiv symbol(%2)) {
      %4 = loom.alloc [%0, %1, %2] on @L1 : memref<?x?x?xf16>
      %5 = loom.semaphore_take %4 : memref<?x?x?xf16> -> memref<?x?x?xf16>
      %6 = loom.init_tensor %5[%0, %1, %2] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
      %7 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
      %8 = loom.alloc [%0, %1, %3] on @L1 : memref<?x?x?xf16>
      %9 = loom.semaphore_take %8 : memref<?x?x?xf16> -> memref<?x?x?xf16>
      %10 = loom.alloc [%0, %3, %2] on @L1 : memref<?x?x?xf16>
      %11 = loom.semaphore_take %10 : memref<?x?x?xf16> -> memref<?x?x?xf16>
      %12 = affine.for %arg6 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%3] iter_args(%arg7 = %7) -> (tensor<?x?x?xf16>) {
        %18 = arith.muli %arg3, %0 : index
        %19 = arith.muli %arg4, %1 : index
        %20 = arith.muli %arg6, %3 : index
        %21 = loom.subview %arg0[%18, %19, %20] [%0, %1, %3] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
        loom.copy %21, %9 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to memref<?x?x?xf16>
        %22 = loom.bufferize_to_tensor %9[%0, %1, %3] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
        %23 = arith.muli %arg5, %2 : index
        %24 = loom.subview %arg1[%18, %20, %23] [%0, %3, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
        loom.copy %24, %11 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to memref<?x?x?xf16>
        %25 = loom.bufferize_to_tensor %11[%0, %3, %2] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
        %26 = linalg.batch_matmul ins(%22, %25 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%arg7 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        loom.semaphore_give %11 : memref<?x?x?xf16>
        loom.semaphore_give %9 : memref<?x?x?xf16>
        affine.yield %26 : tensor<?x?x?xf16>
      }
      %13 = arith.muli %arg3, %0 : index
      %14 = arith.muli %arg4, %1 : index
      %15 = arith.muli %arg5, %2 : index
      %16 = loom.subview %arg2[%13, %14, %15] [%0, %1, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
      %17 = loom.bufferize_to_memref %12 : tensor<?x?x?xf16> -> memref<?x?x?xf16>
      loom.copy %17, %16 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
      loom.semaphore_give %5 : memref<?x?x?xf16>
    }
    return
  }
}
