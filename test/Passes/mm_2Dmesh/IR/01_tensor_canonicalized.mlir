module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  func.func @matmul(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = loom.sym @block_size_0 : index
    %1 = loom.sym @block_size_1 : index
    %2 = loom.sym @block_size_2 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (4096 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
      %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %5 = affine.for %arg5 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%2] iter_args(%arg6 = %4) -> (tensor<?x?xf32>) {
        %9 = arith.muli %arg3, %0 : index
        %10 = arith.muli %arg5, %2 : index
        %subview_0 = memref.subview %arg0[%9, %10] [%0, %2] [1, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %11 = bufferization.to_tensor %subview_0 : memref<?x?xf32, strided<[512, 1], offset: ?>> to tensor<?x?xf32>
        %12 = arith.muli %arg5, %2 : index
        %13 = arith.muli %arg4, %1 : index
        %subview_1 = memref.subview %arg1[%12, %13] [%2, %1] [1, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
        %14 = bufferization.to_tensor %subview_1 : memref<?x?xf32, strided<[4096, 1], offset: ?>> to tensor<?x?xf32>
        %15 = linalg.matmul ins(%11, %14 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg6 : tensor<?x?xf32>) -> tensor<?x?xf32>
        affine.yield %15 : tensor<?x?xf32>
      }
      %6 = arith.muli %arg3, %0 : index
      %7 = arith.muli %arg4, %1 : index
      %subview = memref.subview %arg2[%6, %7] [%0, %1] [1, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      %8 = bufferization.to_buffer %5 : tensor<?x?xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      memref.copy %8, %subview : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
    }
    return
  }
}
