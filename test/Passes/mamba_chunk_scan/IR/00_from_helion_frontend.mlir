#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0, d1) -> ()>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 64 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 8192 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_h, upper_bound = 16 : index} : () -> index
    %4 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 2 : index} : () -> index
    %5 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_c, upper_bound = 8 : index} : () -> index
    %6 = arith.ceildivui %c16, %3 : index
    %7 = arith.ceildivui %c256, %0 : index
    %8 = arith.ceildivui %c64, %1 : index
    %9 = arith.ceildivui %c2, %4 : index
    %10 = arith.ceildivui %c8, %5 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%6), symbol(%7), symbol(%8), symbol(%9), symbol(%10)) {
      %11 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %12 = linalg.fill ins(%cst_1 : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %13 = arith.muli %arg11, %4 : index
      %14 = arith.muli %arg8, %3 : index
      %15 = arith.muli %arg12, %5 : index
      %16 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%13, %14, %15, %16] [1, 1, 1, %0] [1, 1, 1, 1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %17 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %18 = tensor.empty(%0) : tensor<?xf32>
      %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%17 : tensor<?xf16>) outs(%18 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %44 = arith.extf %in : f16 to f32
        linalg.yield %44 : f32
      } -> tensor<?xf32>
      %20 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%19 : tensor<?xf32>) outs(%18 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %44 = arith.truncf %cst_0 : f64 to f32
        %45 = arith.mulf %in, %44 : f32
        linalg.yield %45 : f32
      } -> tensor<?xf32>
      %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%20 : tensor<?xf32>) outs(%18 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %44 = math.powf %cst, %in : f32
        linalg.yield %44 : f32
      } -> tensor<?xf32>
      %22 = arith.muli %15, %c256 : index
      %23 = arith.divui %14, %c16 : index
      %subview_2 = memref.subview %arg4[%13, %22, %23, 0] [1, %0, 1, 16] [1, 1, 1, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %24 = bufferization.to_tensor %subview_2 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %25 = arith.muli %arg10, %1 : index
      %subview_3 = memref.subview %arg5[%13, %15, %14, %25, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %26 = bufferization.to_tensor %subview_3 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %27 = tensor.empty(%1) : tensor<16x?xf16>
      %transposed = linalg.transpose ins(%26 : tensor<?x16xf16>) outs(%27 : tensor<16x?xf16>) permutation = [1, 0] 
      %28 = linalg.matmul ins(%24, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %extracted_slice = tensor.extract_slice %21[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
      %29 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%28, %expanded : tensor<?x?xf32>, tensor<?x1xf32>) outs(%11 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %44 = arith.mulf %in, %in_7 : f32
        linalg.yield %44 : f32
      } -> tensor<?x?xf32>
      %30 = arith.addi %arg9, %c1 : index
      %31 = arith.muli %30, %0 : index
      %32 = arith.ceildivui %31, %2 : index
      %33 = scf.for %arg13 = %c0 to %32 step %c1 iter_args(%arg14 = %29) -> (tensor<?x?xf32>) {
        %44 = arith.muli %arg13, %2 : index
        %subview_7 = memref.subview %arg0[%13, %15, %23, %16, %44] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %45 = bufferization.to_tensor %subview_7 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_8 = memref.subview %arg1[%13, %14, %15, %44] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %46 = bufferization.to_tensor %subview_8 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %47 = tensor.empty(%2) : tensor<?xf32>
        %48 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%46 : tensor<?xf16>) outs(%47 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %64 = arith.extf %in : f16 to f32
          linalg.yield %64 : f32
        } -> tensor<?xf32>
        %extracted_slice_9 = tensor.extract_slice %19[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_10 = tensor.expand_shape %extracted_slice_9 [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
        %49 = tensor.empty(%0) : tensor<?x1xf32>
        %50 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_10 : tensor<?x1xf32>) outs(%49 : tensor<?x1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %64 = arith.truncf %cst_0 : f64 to f32
          %65 = arith.mulf %in, %64 : f32
          linalg.yield %65 : f32
        } -> tensor<?x1xf32>
        %extracted_slice_11 = tensor.extract_slice %48[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_12 = tensor.expand_shape %extracted_slice_11 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %51 = tensor.empty(%2) : tensor<1x?xf32>
        %52 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_12 : tensor<1x?xf32>) outs(%51 : tensor<1x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %64 = arith.truncf %cst_0 : f64 to f32
          %65 = arith.mulf %in, %64 : f32
          linalg.yield %65 : f32
        } -> tensor<1x?xf32>
        %53 = tensor.empty(%0, %2) : tensor<?x?xf32>
        %54 = linalg.generic {indexing_maps = [#map2, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%50, %52 : tensor<?x1xf32>, tensor<1x?xf32>) outs(%53 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %64 = arith.subf %in, %in_17 : f32
          linalg.yield %64 : f32
        } -> tensor<?x?xf32>
        %55 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%54 : tensor<?x?xf32>) outs(%53 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %64 = math.powf %cst, %in : f32
          linalg.yield %64 : f32
        } -> tensor<?x?xf32>
        %56 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%45, %55 : tensor<?x?xf16>, tensor<?x?xf32>) outs(%53 : tensor<?x?xf32>) {
        ^bb0(%in: f16, %in_17: f32, %out: f32):
          %64 = arith.extf %in : f16 to f32
          %65 = arith.mulf %64, %in_17 : f32
          linalg.yield %65 : f32
        } -> tensor<?x?xf32>
        %subview_13 = memref.subview %arg2[%13, %14, %15, %44] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %57 = bufferization.to_tensor %subview_13 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %58 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%57 : tensor<?xf16>) outs(%47 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %64 = arith.extf %in : f16 to f32
          linalg.yield %64 : f32
        } -> tensor<?xf32>
        %extracted_slice_14 = tensor.extract_slice %58[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_15 = tensor.expand_shape %extracted_slice_14 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %59 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%56, %expanded_15 : tensor<?x?xf32>, tensor<1x?xf32>) outs(%53 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %64 = arith.mulf %in, %in_17 : f32
          linalg.yield %64 : f32
        } -> tensor<?x?xf32>
        %60 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %61 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%59 : tensor<?x?xf32>) outs(%60 : tensor<?x?xf16>) {
        ^bb0(%in: f32, %out: f16):
          %64 = arith.truncf %in : f32 to f16
          linalg.yield %64 : f16
        } -> tensor<?x?xf16>
        %subview_16 = memref.subview %arg3[%13, %22, %14, %25] [1, %2, 1, %1] [1, 1, 1, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %62 = bufferization.to_tensor %subview_16 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
        %63 = linalg.matmul ins(%61, %62 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        scf.yield %63 : tensor<?x?xf32>
      }
      %subview_4 = memref.subview %arg6[%14] [1] [1] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      %34 = bufferization.to_tensor %subview_4 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %35 = tensor.empty() : tensor<f32>
      %36 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = []} ins(%34 : tensor<f16>) outs(%35 : tensor<f32>) {
      ^bb0(%in: f16, %out: f32):
        %44 = arith.extf %in : f16 to f32
        linalg.yield %44 : f32
      } -> tensor<f32>
      %subview_5 = memref.subview %arg3[%13, %22, %14, %25] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %37 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
      %38 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%37 : tensor<?x?xf16>) outs(%11 : tensor<?x?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %44 = arith.extf %in : f16 to f32
        linalg.yield %44 : f32
      } -> tensor<?x?xf32>
      %39 = linalg.generic {indexing_maps = [#map1, #map5, #map1], iterator_types = ["parallel", "parallel"]} ins(%38, %36 : tensor<?x?xf32>, tensor<f32>) outs(%11 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %44 = arith.mulf %in, %in_7 : f32
        linalg.yield %44 : f32
      } -> tensor<?x?xf32>
      %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%33, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %44 = arith.addf %in, %in_7 : f32
        linalg.yield %44 : f32
      } -> tensor<?x?xf32>
      %41 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %42 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%40 : tensor<?x?xf32>) outs(%41 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %out: f16):
        %44 = arith.truncf %in : f32 to f16
        linalg.yield %44 : f16
      } -> tensor<?x?xf16>
      %subview_6 = memref.subview %arg7[%13, %22, %14, %25] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %43 = bufferization.to_buffer %42 : tensor<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      memref.copy %43, %subview_6 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    }
    return
  }
}