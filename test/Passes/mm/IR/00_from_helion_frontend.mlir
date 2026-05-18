module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 2048 : index}, loom.tile_n = {is_reduction = false, upper_bound = 256 : index}} {
  func.func @matmul(%x_arg: memref<2048x256xf16>, %y_arg: memref<256x256xf16>, %out__arg: memref<2048x256xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 2048 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 256 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 256 : index} : () -> index
    %3 = arith.ceildivui %c2048, %0 : index
    %4 = arith.ceildivui %c256, %1 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %7 = arith.ceildivui %c256, %2 : index
      %8 = scf.for %arg5 = %c0 to %7 step %c1 iter_args(%arg6 = %6) -> (tensor<?x?xf16>) {
        %12 = arith.muli %arg3, %0 : index
        %13 = arith.muli %arg5, %2 : index
        %subview_0 = memref.subview %x_arg[%12, %13] [%0, %2] [1, 1] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %14 = bufferization.to_tensor %subview_0 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %15 = arith.muli %arg4, %1 : index
        %subview_1 = memref.subview %y_arg[%13, %15] [%2, %1] [1, 1] : memref<256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %16 = bufferization.to_tensor %subview_1 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %17 = linalg.matmul ins(%14, %16 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg6 : tensor<?x?xf16>) -> tensor<?x?xf16>
        scf.yield %17 : tensor<?x?xf16>
      }
      %9 = arith.muli %arg3, %0 : index
      %10 = arith.muli %arg4, %1 : index
      %subview = memref.subview %out__arg[%9, %10] [%0, %1] [1, 1] : memref<2048x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      %11 = bufferization.to_buffer %8 : tensor<?x?xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
      memref.copy %11, %subview : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, strided<[256, 1], offset: ?>>
    }
    return
  }
}