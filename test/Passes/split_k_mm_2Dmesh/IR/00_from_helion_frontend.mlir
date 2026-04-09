module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @split_k_matmul(%arg0: memref<256x256xf32>, %arg1: memref<256x4096xf32>, %arg2: memref<4096x256xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 256 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 4096 : index} : () -> index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (256 ceildiv symbol(%0), 256 ceildiv symbol(%1), 4096 ceildiv symbol(%2)) {
      %3 = arith.muli %arg3, %0 : index
      %6 = arith.muli %arg4, %1 : index
      %4 = arith.muli %arg5, %2 : index
      %subview = memref.subview %arg1[%3, %4] [%0, %2] [1, 1] : memref<256x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      %5 = bufferization.to_tensor %subview : memref<?x?xf32, strided<[4096, 1], offset: ?>> to tensor<?x?xf32>
      %subview_0 = memref.subview %arg2[%4, %6] [%2, %1] [1, 1] : memref<4096x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %7 = bufferization.to_tensor %subview_0 : memref<?x?xf32, strided<[256, 1], offset: ?>> to tensor<?x?xf32>
      %8 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %10 = linalg.matmul ins(%5, %7 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %subview_1 = memref.subview %arg0[%3, %6] [%0, %1] [1, 1] : memref<256x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      %is_first_k = arith.cmpi eq, %4, %c0 : index
      scf.if %is_first_k {
        %11 = loom.reduce_sum %10 (UB : [%c0, %c0], LB : [%c0, %c0]) : tensor<?x?xf32> -> tensor<?x?xf32>
        %12 = bufferization.to_buffer %11 : tensor<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        memref.copy %12, %subview_1 : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      }
    }
    return
  }
}
