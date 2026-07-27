#map = affine_map<()[s0] -> (512 ceildiv s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 8 : index}, loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @batch_matmul(%arg0: memref<8x4096x512xf16>, %arg1: memref<8x512x4096xf16>, %arg2: memref<8x4096x4096xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 8 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 4096 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 4096 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 512 : index} : () -> index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (8 ceildiv symbol(%0), 4096 ceildiv symbol(%1), 4096 ceildiv symbol(%2)) {
      %4 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
      %5 = linalg.fill ins(%cst : f16) outs(%4 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
      %6 = affine.for %arg6 = 0 to #map()[%3] iter_args(%arg7 = %5) -> (tensor<?x?x?xf16>) {
        %11 = arith.muli %arg3, %0 : index
        %12 = arith.muli %arg4, %1 : index
        %13 = arith.muli %arg6, %3 : index
        %subview_0 = memref.subview %arg0[%11, %12, %13] [%0, %1, %3] [1, 1, 1] : memref<8x4096x512xf16> to memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>>
        %14 = bufferization.to_tensor %subview_0 : memref<?x?x?xf16, strided<[2097152, 512, 1], offset: ?>> to tensor<?x?x?xf16>
        %15 = arith.muli %arg5, %2 : index
        %subview_1 = memref.subview %arg1[%11, %13, %15] [%0, %3, %2] [1, 1, 1] : memref<8x512x4096xf16> to memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>>
        %16 = bufferization.to_tensor %subview_1 : memref<?x?x?xf16, strided<[2097152, 4096, 1], offset: ?>> to tensor<?x?x?xf16>
        %17 = arith.index_cast %0 : index to i64
        %18 = arith.cmpi eq, %17, %17 : i64
        cf.assert %18, "mismatching contracting dimension"
        %19 = arith.index_cast %3 : index to i64
        %20 = arith.cmpi eq, %19, %19 : i64
        cf.assert %20, "mismatching contracting dimension"
        %21 = linalg.batch_matmul ins(%14, %16 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%5 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %22 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21, %arg7 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%4 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %23 = arith.addf %in, %in_2 : f16
          linalg.yield %23 : f16
        } -> tensor<?x?x?xf16>
        affine.yield %22 : tensor<?x?x?xf16>
      }
      %7 = arith.muli %arg3, %0 : index
      %8 = arith.muli %arg4, %1 : index
      %9 = arith.muli %arg5, %2 : index
      %subview = memref.subview %arg2[%7, %8, %9] [%0, %1, %2] [1, 1, 1] : memref<8x4096x4096xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
      %10 = bufferization.to_buffer %6 : tensor<?x?x?xf16> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
      memref.copy %10, %subview : memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>> to memref<?x?x?xf16, strided<[16777216, 4096, 1], offset: ?>>
    }
    return
  }
}