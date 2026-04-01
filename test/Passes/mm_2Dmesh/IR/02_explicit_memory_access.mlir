module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  func.func @_matmul(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = loom.sym @tile_m {upper_bound = 4096 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 4096 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 512 : index} : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (4096 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
      %3 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %4 = loom.semaphore_take %3 : memref<?x?xf16> -> memref<?x?xf16>
      %5 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %6 = loom.semaphore_take %5 : memref<?x?xf16> -> memref<?x?xf16>
      %7 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %8 = loom.semaphore_take %7 : memref<?x?xf16> -> memref<?x?xf16>
      %9 = loom.init_tensor %8[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %10 = linalg.fill ins(%cst : f16) outs(%9 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %11 = affine.for %arg5 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%2] iter_args(%arg6 = %10) -> (tensor<?x?xf16>) {
        %16 = arith.muli %arg3, %0 : index
        %17 = arith.muli %arg5, %2 : index
        %18 = loom.subview %arg0[%16, %17] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        loom.copy %18, %6 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
        %19 = loom.bufferize_to_tensor %6[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %20 = arith.muli %arg4, %1 : index
        %21 = loom.subview %arg1[%17, %20] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        loom.copy %21, %4 src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
        %22 = loom.bufferize_to_tensor %4[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %23 = linalg.matmul ins(%19, %22 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg6 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %4 : memref<?x?xf16>
        loom.semaphore_give %6 : memref<?x?xf16>
        affine.yield %23 : tensor<?x?xf16>
      }
      %12 = arith.muli %arg3, %0 : index
      %13 = arith.muli %arg4, %1 : index
      %14 = loom.subview %arg2[%12, %13] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %15 = loom.bufferize_to_memref %11 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %15, %14 src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %8 : memref<?x?xf16>
    }
    return
  }
}
