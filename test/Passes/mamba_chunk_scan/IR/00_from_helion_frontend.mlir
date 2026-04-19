#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
#map4 = affine_map<(d0, d1) -> ()>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.442380e+00 : f16
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 64 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 8192 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_h, upper_bound = 64 : index} : () -> index
    %4 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 2 : index} : () -> index
    %5 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_c, upper_bound = 8 : index} : () -> index
    %6 = arith.ceildivui %c64, %3 : index
    %7 = arith.ceildivui %c256, %0 : index
    %8 = arith.ceildivui %c64, %1 : index
    %9 = arith.ceildivui %c2, %4 : index
    %10 = arith.ceildivui %c8, %5 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%6), symbol(%7), symbol(%8), symbol(%9), symbol(%10)) {
      %11 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %12 = linalg.fill ins(%cst_0 : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %13 = arith.muli %arg11, %4 : index
      %14 = arith.muli %arg8, %3 : index
      %15 = arith.muli %arg12, %5 : index
      %16 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%13, %14, %15, %16] [1, 1, 1, %0] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %17 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %18 = tensor.empty(%0) : tensor<?xf16>
      %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%17 : tensor<?xf16>) outs(%18 : tensor<?xf16>) {
      ^bb0(%in: f16, %out: f16):
        %38 = arith.mulf %in, %cst_1 : f16
        linalg.yield %38 : f16
      } -> tensor<?xf16>
      %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%19 : tensor<?xf16>) outs(%18 : tensor<?xf16>) {
      ^bb0(%in: f16, %out: f16):
        %38 = math.powf %cst, %in : f16
        linalg.yield %38 : f16
      } -> tensor<?xf16>
      %21 = arith.muli %15, %c256 : index
      %22 = arith.divui %14, %c64 : index
      %subview_2 = memref.subview %arg4[%13, %21, %22, 0] [1, %0, 1, 64] [1, 1, 1, 1] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %23 = bufferization.to_tensor %subview_2 : memref<?x64xf16, strided<[64, 1], offset: ?>> to tensor<?x64xf16>
      %24 = arith.muli %arg10, %1 : index
      %subview_3 = memref.subview %arg5[%13, %15, %14, %24, 0] [1, 1, 1, %1, 64] [1, 1, 1, 1, 1] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %25 = bufferization.to_tensor %subview_3 : memref<?x64xf16, strided<[64, 1], offset: ?>> to tensor<?x64xf16>
      %26 = tensor.empty(%1) : tensor<64x?xf16>
      %transposed = linalg.transpose ins(%25 : tensor<?x64xf16>) outs(%26 : tensor<64x?xf16>) permutation = [1, 0] 
      %27 = linalg.matmul ins(%23, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%12 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %extracted_slice = tensor.extract_slice %20[0] [%0] [1] : tensor<?xf16> to tensor<?xf16>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [%0, 1] : tensor<?xf16> into tensor<?x1xf16>
      %28 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%27, %expanded : tensor<?x?xf16>, tensor<?x1xf16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_7: f16, %out: f16):
        %38 = arith.mulf %in, %in_7 : f16
        linalg.yield %38 : f16
      } -> tensor<?x?xf16>
      %29 = arith.addi %arg9, %c1 : index
      %30 = arith.muli %29, %0 : index
      %31 = arith.ceildivui %30, %2 : index
      %32 = scf.for %arg13 = %c0 to %31 step %c1 iter_args(%arg14 = %28) -> (tensor<?x?xf16>) {
        %38 = arith.muli %arg13, %2 : index
        %subview_7 = memref.subview %arg0[%13, %15, %22, %16, %38] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %39 = bufferization.to_tensor %subview_7 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_8 = memref.subview %arg1[%13, %14, %15, %38] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %40 = bufferization.to_tensor %subview_8 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %extracted_slice_9 = tensor.extract_slice %17[0] [%0] [1] : tensor<?xf16> to tensor<?xf16>
        %expanded_10 = tensor.expand_shape %extracted_slice_9 [[0, 1]] output_shape [%0, 1] : tensor<?xf16> into tensor<?x1xf16>
        %41 = tensor.empty(%0) : tensor<?x1xf16>
        %42 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_10 : tensor<?x1xf16>) outs(%41 : tensor<?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.mulf %in, %cst_1 : f16
          linalg.yield %53 : f16
        } -> tensor<?x1xf16>
        %extracted_slice_11 = tensor.extract_slice %40[0] [%2] [1] : tensor<?xf16> to tensor<?xf16>
        %expanded_12 = tensor.expand_shape %extracted_slice_11 [[0, 1]] output_shape [1, %2] : tensor<?xf16> into tensor<1x?xf16>
        %43 = tensor.empty(%2) : tensor<1x?xf16>
        %44 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_12 : tensor<1x?xf16>) outs(%43 : tensor<1x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.mulf %in, %cst_1 : f16
          linalg.yield %53 : f16
        } -> tensor<1x?xf16>
        %45 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %46 = linalg.generic {indexing_maps = [#map2, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%42, %44 : tensor<?x1xf16>, tensor<1x?xf16>) outs(%45 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_17: f16, %out: f16):
          %53 = arith.subf %in, %in_17 : f16
          linalg.yield %53 : f16
        } -> tensor<?x?xf16>
        %47 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%46 : tensor<?x?xf16>) outs(%45 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = math.powf %cst, %in : f16
          linalg.yield %53 : f16
        } -> tensor<?x?xf16>
        %48 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%39, %47 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%45 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_17: f16, %out: f16):
          %53 = arith.mulf %in, %in_17 : f16
          linalg.yield %53 : f16
        } -> tensor<?x?xf16>
        %subview_13 = memref.subview %arg2[%13, %14, %15, %38] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %49 = bufferization.to_tensor %subview_13 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %extracted_slice_14 = tensor.extract_slice %49[0] [%2] [1] : tensor<?xf16> to tensor<?xf16>
        %expanded_15 = tensor.expand_shape %extracted_slice_14 [[0, 1]] output_shape [1, %2] : tensor<?xf16> into tensor<1x?xf16>
        %50 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%48, %expanded_15 : tensor<?x?xf16>, tensor<1x?xf16>) outs(%45 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_17: f16, %out: f16):
          %53 = arith.mulf %in, %in_17 : f16
          linalg.yield %53 : f16
        } -> tensor<?x?xf16>
        %subview_16 = memref.subview %arg3[%13, %21, %14, %24] [1, %2, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        %51 = bufferization.to_tensor %subview_16 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
        %52 = linalg.matmul ins(%50, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) -> tensor<?x?xf16>
        scf.yield %52 : tensor<?x?xf16>
      }
      %subview_4 = memref.subview %arg6[%14] [1] [1] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      %33 = bufferization.to_tensor %subview_4 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %subview_5 = memref.subview %arg3[%13, %21, %14, %24] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %34 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
      %35 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel"]} ins(%34, %33 : tensor<?x?xf16>, tensor<f16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_7: f16, %out: f16):
        %38 = arith.mulf %in, %in_7 : f16
        linalg.yield %38 : f16
      } -> tensor<?x?xf16>
      %36 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%32, %35 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_7: f16, %out: f16):
        %38 = arith.addf %in, %in_7 : f16
        linalg.yield %38 : f16
      } -> tensor<?x?xf16>
      %subview_6 = memref.subview %arg7[%13, %21, %14, %24] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %37 = bufferization.to_buffer %36 : tensor<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      memref.copy %37, %subview_6 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
    }
    return
  }
}