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
      %5 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %6 = arith.ceildivui %c256, %2 : index
      %7 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %8 = scf.for %arg5 = %c0 to %6 step %c1 iter_args(%arg6 = %7) -> (tensor<?x?xf16>) {
        %14 = arith.muli %arg3, %0 : index
        %15 = arith.muli %arg5, %2 : index
        %subview_0 = memref.subview %arg0[%14, %15] [%0, %2] [1, 1] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %16 = loom.bufferize_to_tensor %subview_0[%0, %2] : memref<?x?xf16, strided<[256, 1], offset: ?>> -> tensor<?x?xf16>
        %17 = arith.muli %arg4, %1 : index
        %subview_1 = memref.subview %arg1[%15, %17] [%2, %1] [1, 1] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %18 = loom.bufferize_to_tensor %subview_1[%2, %1] : memref<?x?xf16, strided<[256, 1], offset: ?>> -> tensor<?x?xf16>
        %19 = linalg.matmul ins(%16, %18 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg6 : tensor<?x?xf16>) -> tensor<?x?xf16>
        scf.yield %19 : tensor<?x?xf16>
      }
      %9 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %10 = linalg.copy ins(%8 : tensor<?x?xf16>) outs(%9 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %11 = arith.muli %arg3, %0 : index
      %12 = arith.muli %arg4, %1 : index
      %subview = memref.subview %arg2[%11, %12] [%0, %1] [1, 1] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      %13 = loom.bufferize_to_memref %10 : tensor<?x?xf16> -> memref<?x?xf16, strided<[256, 1], offset: ?>>
      memref.copy %13, %subview : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
    }
    return
  }
}
