module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @split_k_matmul(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 256 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 4096 : index} : () -> index
    %3 = arith.ceildivui %c256, %0 : index
    %4 = arith.ceildivui %c256, %1 : index
    %5 = arith.ceildivui %c4096, %2 : index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%3), symbol(%4), symbol(%5)) {
      %6 = arith.muli %arg3, %0 : index
      %7 = arith.muli %arg5, %2 : index
      %subview = memref.subview %arg1[%6, %7] [%0, %2] [1, 1] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      %8 = bufferization.to_tensor %subview : memref<?x?xf32, strided<[4096, 1], offset: ?>> to tensor<?x?xf32>
      %9 = arith.muli %arg4, %1 : index
      %subview_0 = memref.subview %arg2[%7, %9] [%2, %1] [1, 1] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %10 = bufferization.to_tensor %subview_0 : memref<?x?xf32, strided<[256, 1], offset: ?>> to tensor<?x?xf32>
      %11 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %12 = linalg.fill ins(%cst : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %13 = linalg.matmul ins(%8, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %14 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %14 {
        %subview_1 = memref.subview %arg0[%6, %9] [%0, %1] [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        %cst_1 = arith.constant 0.000000e+00 : f32
        %result = tensor.empty(%0, %1) : tensor<?x?xf32>
        %filled = linalg.fill ins(%cst_1 : f32) outs(%result : tensor<?x?xf32>) -> tensor<?x?xf32>
        %15 = loom.reduce_sum ins(%13) outs(%filled) (UL : [%c0, %c0], LR : [%c0, %c0]) : tensor<?x?xf32> -> tensor<?x?xf32>
        %16 = bufferization.to_buffer %15 : tensor<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        memref.copy %16, %subview_1 : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      }
    }
    return
  }
}