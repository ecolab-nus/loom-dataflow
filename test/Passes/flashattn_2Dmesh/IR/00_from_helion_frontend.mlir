#map = affine_map<()[s0] -> (512 ceildiv s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_3 = -1 : index} {
  func.func @attention(%arg0: memref<8x128x512xf16>, %arg1: memref<8x512x128xf16>, %arg2: memref<8x512x128xf16>, %arg3: memref<8x512x128xf16>) {
    %cst = arith.constant 2.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %cst_2 = arith.constant 0xFC00 : f16
    %cst_3 = arith.constant 1.275630e-01 : f16
    %0 = "loom.sym"() {symbol_ref = @block_size_0} : () -> index
    %1 = "loom.sym"() {symbol_ref = @block_size_1} : () -> index
    %2 = "loom.sym"() {symbol_ref = @block_size_3} : () -> index
    affine.parallel (%arg4, %arg5) = (0, 0) to (8 ceildiv symbol(%0), 512 ceildiv symbol(%1)) {
      %3 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %4 = linalg.fill ins(%cst_2 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %5 = linalg.fill ins(%cst_1 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %6 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
      %7 = linalg.fill ins(%cst_0 : f16) outs(%6 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %8 = arith.muli %arg4, %0 : index
      %9 = arith.muli %arg5, %1 : index
      %subview = memref.subview %arg2[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<8x512x128xf16> to memref<?x?x128xf16, strided<[65536, 128, 1], offset: ?>>
      %10 = bufferization.to_tensor %subview : memref<?x?x128xf16, strided<[65536, 128, 1], offset: ?>> to tensor<?x?x128xf16>
      %11:3 = affine.for %arg6 = 0 to #map()[%2] iter_args(%arg7 = %4, %arg8 = %5, %arg9 = %7) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
        %14 = arith.muli %arg6, %2 : index
        %subview_5 = memref.subview %arg0[%8, 0, %14] [%0, 128, %2] [1, 1, 1] : memref<8x128x512xf16> to memref<?x128x?xf16, strided<[65536, 512, 1], offset: ?>>
        %15 = bufferization.to_tensor %subview_5 : memref<?x128x?xf16, strided<[65536, 512, 1], offset: ?>> to tensor<?x128x?xf16>
        %16 = arith.index_cast %0 : index to i64
        %17 = arith.cmpi eq, %16, %16 : i64
        cf.assert %17, "mismatching contracting dimension"
        %18 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
        %19 = linalg.fill ins(%cst_0 : f16) outs(%18 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %20 = linalg.batch_matmul ins(%10, %15 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%19 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %21 = tensor.empty(%0, %1) : tensor<?x?xi64>
        %22 = linalg.fill ins(%c0_i64 : i64) outs(%21 : tensor<?x?xi64>) -> tensor<?x?xi64>
        %23:2 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%20 : tensor<?x?x?xf16>) outs(%4, %22 : tensor<?x?xf16>, tensor<?x?xi64>) {
        ^bb0(%in: f16, %out: f16, %out_11: i64):
          %41 = linalg.index 2 : index
          %42 = arith.index_cast %41 : index to i64
          %43 = arith.maximumf %in, %out : f16
          %44 = arith.cmpf ogt, %in, %out : f16
          %45 = arith.select %44, %42, %out_11 : i64
          linalg.yield %43, %45 : f16, i64
        } -> (tensor<?x?xf16>, tensor<?x?xi64>)
        %24 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%23#0 : tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %41 = arith.mulf %in, %cst_3 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?xf16>
        %25 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg7, %24 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %41 = arith.cmpf ogt, %in, %in_11 : f16
          %42 = arith.select %41, %in, %in_11 : f16
          linalg.yield %42 : f16
        } -> tensor<?x?xf16>
        %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<?x?x?xf16>) outs(%18 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %41 = arith.mulf %in, %cst_3 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?x?xf16>
        %extracted_slice_6 = tensor.extract_slice %25[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %expanded_7 = tensor.expand_shape %extracted_slice_6 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
        %27 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %expanded_7 : tensor<?x?x?xf16>, tensor<?x?x1xf16>) outs(%18 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %41 = arith.subf %in, %in_11 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?x?xf16>
        %28 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27 : tensor<?x?x?xf16>) outs(%18 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %41 = math.powf %cst, %in : f16
          linalg.yield %41 : f16
        } -> tensor<?x?x?xf16>
        %29 = linalg.fill ins(%cst_0 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %30 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28 : tensor<?x?x?xf16>) outs(%29 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %41 = arith.addf %in, %out : f16
          linalg.yield %41 : f16
        } -> tensor<?x?xf16>
        %31 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg7, %25 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %41 = arith.subf %in, %in_11 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?xf16>
        %32 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%31 : tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %41 = math.powf %cst, %in : f16
          linalg.yield %41 : f16
        } -> tensor<?x?xf16>
        %33 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg8, %32 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %41 = arith.mulf %in, %in_11 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?xf16>
        %34 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%33, %30 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %41 = arith.addf %in, %in_11 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?xf16>
        %extracted_slice_8 = tensor.extract_slice %32[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %expanded_9 = tensor.expand_shape %extracted_slice_8 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
        %35 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %expanded_9 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%6 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %41 = arith.mulf %in, %in_11 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?x128xf16>
        %subview_10 = memref.subview %arg1[%8, %14, 0] [%0, %2, 128] [1, 1, 1] : memref<8x512x128xf16> to memref<?x?x128xf16, strided<[65536, 128, 1], offset: ?>>
        %36 = bufferization.to_tensor %subview_10 : memref<?x?x128xf16, strided<[65536, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %17, "mismatching contracting dimension"
        %37 = arith.index_cast %2 : index to i64
        %38 = arith.cmpi eq, %37, %37 : i64
        cf.assert %38, "mismatching contracting dimension"
        %39 = linalg.batch_matmul ins(%28, %36 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%7 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %35 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%6 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %41 = arith.addf %in, %in_11 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?x128xf16>
        affine.yield %25, %34, %40 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
      }
      %extracted_slice = tensor.extract_slice %11#1[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
      %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
      %12 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11#2, %expanded : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%6 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_5: f16, %out: f16):
        %14 = arith.divf %in, %in_5 : f16
        linalg.yield %14 : f16
      } -> tensor<?x?x128xf16>
      %subview_4 = memref.subview %arg3[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<8x512x128xf16> to memref<?x?x128xf16, strided<[65536, 128, 1], offset: ?>>
      %13 = bufferization.to_buffer %12 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[65536, 128, 1], offset: ?>>
      memref.copy %13, %subview_4 : memref<?x?x128xf16, strided<[65536, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[65536, 128, 1], offset: ?>>
    }
    return
  }
}