module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  func.func @matmul(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = loom.sym @block_size_0 : index
    %1 = loom.sym @block_size_1 : index
    %2 = loom.sym @block_size_2 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (4096 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
      %3 = loom.alloc [%2, %1] on @L1 : memref<?x?xf32>
      %4 = loom.semaphore_take %3 : memref<?x?xf32> -> memref<?x?xf32>
      %5 = loom.alloc [%0, %2] on @L1 : memref<?x?xf32>
      %6 = loom.semaphore_take %5 : memref<?x?xf32> -> memref<?x?xf32>
      %7 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
      %8 = loom.semaphore_take %7 : memref<?x?xf32> -> memref<?x?xf32>
      %9 = loom.init_tensor %8[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %11 = affine.for %arg5 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%2] iter_args(%arg6 = %10) -> (tensor<?x?xf32>) {
        %15 = arith.muli %arg3, %0 : index
        %16 = arith.muli %arg5, %2 : index
        %17 = loom.subview %arg0[%15, %16] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %18 = loom.copy_to_tensor %17, %6, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, memref<?x?xf32> -> tensor<?x?xf32>
        %19 = arith.muli %arg5, %2 : index
        %20 = arith.muli %arg4, %1 : index
        %21 = loom.subview %arg1[%19, %20] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
        %22 = loom.copy_to_tensor %21, %4, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, memref<?x?xf32> -> tensor<?x?xf32>
        %23 = linalg.matmul ins(%18, %22 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg6 : tensor<?x?xf32>) -> tensor<?x?xf32>
        loom.semaphore_give %4 : memref<?x?xf32>
        loom.semaphore_give %6 : memref<?x?xf32>
        affine.yield %23 : tensor<?x?xf32>
      }
      %12 = arith.muli %arg3, %0 : index
      %13 = arith.muli %arg4, %1 : index
      %14 = loom.subview %arg2[%12, %13] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      loom.copy_from_tensor %11, %14 : tensor<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %8 : memref<?x?xf32>
    }
    return
  }
}
