#map = affine_map<(d0, d1) -> (d0, d1)>
module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @matmul(%x_arg: memref<4096x512xf16>, %y_arg: memref<512x4096xf16>, %out__arg: memref<4096x4096xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c4096 = arith.constant 4096 : index
    %c512 = arith.constant 512 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 4096 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 4096 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 512 : index} : () -> index
    %3 = arith.ceildivui %c4096, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %7 = arith.ceildivui %c512, %2 : index
      %8 = scf.for %arg5 = %c0 to %7 step %c1 iter_args(%arg6 = %6) -> (tensor<?x?xf16>) {
        %12 = arith.muli %arg3, %0 : index
        %13 = arith.muli %arg5, %2 : index
        %subview_0 = memref.subview %x_arg[%12, %13] [%0, %2] [1, 1] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        %14 = bufferization.to_tensor %subview_0 : memref<?x?xf16, strided<[512, 1], offset: ?>> to tensor<?x?xf16>
        %15 = arith.muli %arg4, %1 : index
        %subview_1 = memref.subview %y_arg[%13, %15] [%2, %1] [1, 1] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        %16 = bufferization.to_tensor %subview_1 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
        %17 = linalg.matmul ins(%14, %16 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%6 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg6, %17 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %19 = arith.addf %in, %in_2 : f16
          linalg.yield %19 : f16
        } -> tensor<?x?xf16>
        scf.yield %18 : tensor<?x?xf16>
      }
      %9 = arith.muli %arg3, %0 : index
      %10 = arith.muli %arg4, %1 : index
      %subview = memref.subview %out__arg[%9, %10] [%0, %1] [1, 1] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %11 = bufferization.to_buffer %8 : tensor<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      memref.copy %11, %subview : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
    }
    return
  }
}