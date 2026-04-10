module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @split_k_matmul(%arg0: memref<256x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x256xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
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
      %subview = memref.subview %arg1[%6, %7] [%0, %2] [1, 1] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %8 = bufferization.to_tensor %subview : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
      %9 = arith.muli %arg4, %1 : index
      %subview_0 = memref.subview %arg2[%7, %9] [%2, %1] [1, 1] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      %10 = bufferization.to_tensor %subview_0 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
      %11 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %12 = linalg.fill ins(%cst : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %13 = linalg.matmul ins(%8, %10 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%12 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %14 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %14 {
        %subview_1 = memref.subview %arg0[%6, %9] [%0, %1] [1, 1] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %15 = tensor.empty(%0, %1) : tensor<?x?xf16>
        %16 = linalg.fill ins(%cst : f16) outs(%15 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %17 = loom.reduce_sum ins(%13) outs(%16) (UL : [%c0, %c0], LR : [%c0, %c0]) : tensor<?x?xf16> -> tensor<?x?xf16>
        %18 = bufferization.to_buffer %17 : tensor<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        memref.copy %18, %subview_1 : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      }
    }
    return
  }
}
