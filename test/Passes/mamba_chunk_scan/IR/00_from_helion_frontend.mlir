#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#map3 = affine_map<(d0, d1) -> (0, d1)>
#map4 = affine_map<() -> ()>
#map5 = affine_map<(d0, d1) -> ()>
module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 64 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 8192 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_h, upper_bound = 16 : index} : () -> index
    %4 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_c, upper_bound = 8 : index} : () -> index
    %5 = arith.ceildivui %c16, %3 : index
    %6 = arith.ceildivui %c256, %0 : index
    %7 = arith.ceildivui %c64, %1 : index
    %8 = arith.ceildivui %c8, %4 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%5), symbol(%6), symbol(%7), 1, symbol(%8)) {
      %9 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %10 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %11 = arith.muli %arg8, %3 : index
      %12 = arith.muli %arg12, %4 : index
      %13 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%arg11, %11, %12, %13] [1, 1, 1, %0] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %14 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %15 = tensor.empty(%0) : tensor<?xf32>
      %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%14 : tensor<?xf16>) outs(%15 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %41 = arith.extf %in : f16 to f32
        linalg.yield %41 : f32
      } -> tensor<?xf32>
      %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%16 : tensor<?xf32>) outs(%15 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %41 = arith.truncf %cst_0 : f64 to f32
        %42 = arith.mulf %in, %41 : f32
        linalg.yield %42 : f32
      } -> tensor<?xf32>
      %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%17 : tensor<?xf32>) outs(%15 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %41 = math.powf %cst, %in : f32
        linalg.yield %41 : f32
      } -> tensor<?xf32>
      %19 = arith.muli %12, %c256 : index
      %20 = arith.divui %11, %c16 : index
      %subview_2 = memref.subview %arg4[%arg11, %19, %20, 0] [1, %0, 1, 16] [1, 1, 1, 1] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %21 = bufferization.to_tensor %subview_2 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %22 = arith.muli %arg10, %1 : index
      %subview_3 = memref.subview %arg5[%arg11, %12, %11, %22, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %23 = bufferization.to_tensor %subview_3 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %24 = tensor.empty(%1) : tensor<16x?xf16>
      %transposed = linalg.transpose ins(%23 : tensor<?x16xf16>) outs(%24 : tensor<16x?xf16>) permutation = [1, 0] 
      %25 = linalg.matmul ins(%21, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %extracted_slice = tensor.extract_slice %18[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
      %26 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%25, %expanded : tensor<?x?xf32>, tensor<?x1xf32>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %41 = arith.mulf %in, %in_7 : f32
        linalg.yield %41 : f32
      } -> tensor<?x?xf32>
      %27 = arith.addi %arg9, %c1 : index
      %28 = arith.muli %27, %0 : index
      %29 = arith.ceildivui %28, %2 : index
      %30 = scf.for %arg13 = %c0 to %29 step %c1 iter_args(%arg14 = %26) -> (tensor<?x?xf32>) {
        %41 = arith.muli %arg13, %2 : index
        %subview_7 = memref.subview %arg0[%arg11, %12, %20, %13, %41] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %42 = bufferization.to_tensor %subview_7 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_8 = memref.subview %arg1[%arg11, %11, %12, %41] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %43 = bufferization.to_tensor %subview_8 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %44 = tensor.empty(%2) : tensor<?xf32>
        %45 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%44 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %61 = arith.extf %in : f16 to f32
          linalg.yield %61 : f32
        } -> tensor<?xf32>
        %extracted_slice_9 = tensor.extract_slice %16[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_10 = tensor.expand_shape %extracted_slice_9 [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
        %46 = tensor.empty(%0) : tensor<?x1xf32>
        %47 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_10 : tensor<?x1xf32>) outs(%46 : tensor<?x1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %61 = arith.truncf %cst_0 : f64 to f32
          %62 = arith.mulf %in, %61 : f32
          linalg.yield %62 : f32
        } -> tensor<?x1xf32>
        %extracted_slice_11 = tensor.extract_slice %45[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_12 = tensor.expand_shape %extracted_slice_11 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %48 = tensor.empty(%2) : tensor<1x?xf32>
        %49 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_12 : tensor<1x?xf32>) outs(%48 : tensor<1x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %61 = arith.truncf %cst_0 : f64 to f32
          %62 = arith.mulf %in, %61 : f32
          linalg.yield %62 : f32
        } -> tensor<1x?xf32>
        %50 = tensor.empty(%0, %2) : tensor<?x?xf32>
        %51 = linalg.generic {indexing_maps = [#map2, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%47, %49 : tensor<?x1xf32>, tensor<1x?xf32>) outs(%50 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %61 = arith.subf %in, %in_17 : f32
          linalg.yield %61 : f32
        } -> tensor<?x?xf32>
        %52 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%51 : tensor<?x?xf32>) outs(%50 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %61 = math.powf %cst, %in : f32
          linalg.yield %61 : f32
        } -> tensor<?x?xf32>
        %53 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%42, %52 : tensor<?x?xf16>, tensor<?x?xf32>) outs(%50 : tensor<?x?xf32>) {
        ^bb0(%in: f16, %in_17: f32, %out: f32):
          %61 = arith.extf %in : f16 to f32
          %62 = arith.mulf %61, %in_17 : f32
          linalg.yield %62 : f32
        } -> tensor<?x?xf32>
        %subview_13 = memref.subview %arg2[%arg11, %11, %12, %41] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %54 = bufferization.to_tensor %subview_13 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %55 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%54 : tensor<?xf16>) outs(%44 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %61 = arith.extf %in : f16 to f32
          linalg.yield %61 : f32
        } -> tensor<?xf32>
        %extracted_slice_14 = tensor.extract_slice %55[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_15 = tensor.expand_shape %extracted_slice_14 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %56 = linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%53, %expanded_15 : tensor<?x?xf32>, tensor<1x?xf32>) outs(%50 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_17: f32, %out: f32):
          %61 = arith.mulf %in, %in_17 : f32
          linalg.yield %61 : f32
        } -> tensor<?x?xf32>
        %57 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %58 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%56 : tensor<?x?xf32>) outs(%57 : tensor<?x?xf16>) {
        ^bb0(%in: f32, %out: f16):
          %61 = arith.truncf %in : f32 to f16
          linalg.yield %61 : f16
        } -> tensor<?x?xf16>
        %subview_16 = memref.subview %arg3[%arg11, %19, %11, %22] [1, %2, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %59 = bufferization.to_tensor %subview_16 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
        %60 = linalg.matmul ins(%58, %59 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        scf.yield %60 : tensor<?x?xf32>
      }
      %subview_4 = memref.subview %arg6[%11] [1] [1] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      %31 = bufferization.to_tensor %subview_4 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %32 = tensor.empty() : tensor<f32>
      %33 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = []} ins(%31 : tensor<f16>) outs(%32 : tensor<f32>) {
      ^bb0(%in: f16, %out: f32):
        %41 = arith.extf %in : f16 to f32
        linalg.yield %41 : f32
      } -> tensor<f32>
      %subview_5 = memref.subview %arg3[%arg11, %19, %11, %22] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %34 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
      %35 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%34 : tensor<?x?xf16>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %41 = arith.extf %in : f16 to f32
        linalg.yield %41 : f32
      } -> tensor<?x?xf32>
      %36 = linalg.generic {indexing_maps = [#map1, #map5, #map1], iterator_types = ["parallel", "parallel"]} ins(%35, %33 : tensor<?x?xf32>, tensor<f32>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %41 = arith.mulf %in, %in_7 : f32
        linalg.yield %41 : f32
      } -> tensor<?x?xf32>
      %37 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%30, %36 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %41 = arith.addf %in, %in_7 : f32
        linalg.yield %41 : f32
      } -> tensor<?x?xf32>
      %38 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %39 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%37 : tensor<?x?xf32>) outs(%38 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %out: f16):
        %41 = arith.truncf %in : f32 to f16
        linalg.yield %41 : f16
      } -> tensor<?x?xf16>
      %subview_6 = memref.subview %arg7[%arg11, %19, %11, %22] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %40 = bufferization.to_buffer %39 : tensor<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      memref.copy %40, %subview_6 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    }
    return
  }
}