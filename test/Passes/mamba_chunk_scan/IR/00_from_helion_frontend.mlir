#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 floordiv 16)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
#map4 = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
#map5 = affine_map<(d0, d1) -> (0, d1)>
#map6 = affine_map<() -> ()>
#map7 = affine_map<(d0, d1) -> ()>
module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x?xf16>, %arg5: memref<1x8x16x64x?xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 256 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 64 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_k, upper_bound = 8192 : index} : () -> index
    %3 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_h, upper_bound = 16 : index} : () -> index
    %4 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_c, upper_bound = 8 : index} : () -> index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (16 ceildiv symbol(%3), 256 ceildiv symbol(%0), 64 ceildiv symbol(%1), 1, 8 ceildiv symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %6 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %7 = arith.muli %arg8, %3 : index
      %8 = arith.muli %arg12, %4 : index
      %9 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%arg11, %7, %8, %9] [1, 1, 1, %0] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %10 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %11 = tensor.empty(%0) : tensor<?xf32>
      %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%10 : tensor<?xf16>) outs(%11 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %37 = arith.extf %in : f16 to f32
        linalg.yield %37 : f32
      } -> tensor<?xf32>
      %13 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%12 : tensor<?xf32>) outs(%11 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %37 = arith.truncf %cst_0 : f64 to f32
        %38 = arith.mulf %in, %37 : f32
        linalg.yield %38 : f32
      } -> tensor<?xf32>
      %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : tensor<?xf32>) outs(%11 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %37 = math.powf %cst, %in : f32
        linalg.yield %37 : f32
      } -> tensor<?xf32>
      %15 = arith.muli %8, %c256 : index
      %16 = affine.apply #map1(%7)
      %dim = memref.dim %arg4, %c3 : memref<1x2048x1x?xf16>
      %subview_2 = memref.subview %arg4[%arg11, %15, %16, 0] [1, %0, 1, %dim] [1, 1, 1, 1] : memref<1x2048x1x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %17 = bufferization.to_tensor %subview_2 : memref<?x?xf16, strided<[?, 1], offset: ?>> to tensor<?x?xf16>
      %18 = arith.muli %arg10, %1 : index
      %dim_3 = memref.dim %arg5, %c4 : memref<1x8x16x64x?xf16>
      %subview_4 = memref.subview %arg5[%arg11, %8, %7, %18, 0] [1, 1, 1, %1, %dim_3] [1, 1, 1, 1, 1] : memref<1x8x16x64x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %19 = bufferization.to_tensor %subview_4 : memref<?x?xf16, strided<[?, 1], offset: ?>> to tensor<?x?xf16>
      %20 = tensor.empty(%dim_3, %1) : tensor<?x?xf16>
      %transposed = linalg.transpose ins(%19 : tensor<?x?xf16>) outs(%20 : tensor<?x?xf16>) permutation = [1, 0] 
      %21 = linalg.matmul ins(%17, %transposed : tensor<?x?xf16>, tensor<?x?xf16>) outs(%6 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %extracted_slice = tensor.extract_slice %14[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
      %expanded = tensor.expand_shape %extracted_slice [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
      %22 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%21, %expanded : tensor<?x?xf32>, tensor<?x1xf32>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %37 = arith.mulf %in, %in_8 : f32
        linalg.yield %37 : f32
      } -> tensor<?x?xf32>
      %23 = arith.addi %arg9, %c1 : index
      %24 = arith.muli %23, %0 : index
      %25 = affine.apply #map4(%24)[%2]
      %26 = scf.for %arg13 = %c0 to %25 step %c1 iter_args(%arg14 = %22) -> (tensor<?x?xf32>) {
        %37 = arith.muli %arg13, %2 : index
        %subview_8 = memref.subview %arg0[%arg11, %8, %16, %9, %37] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %38 = bufferization.to_tensor %subview_8 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_9 = memref.subview %arg1[%arg11, %7, %8, %37] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %39 = bufferization.to_tensor %subview_9 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %40 = tensor.empty(%2) : tensor<?xf32>
        %41 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%39 : tensor<?xf16>) outs(%40 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %57 = arith.extf %in : f16 to f32
          linalg.yield %57 : f32
        } -> tensor<?xf32>
        %extracted_slice_10 = tensor.extract_slice %12[0] [%0] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_11 = tensor.expand_shape %extracted_slice_10 [[0, 1]] output_shape [%0, 1] : tensor<?xf32> into tensor<?x1xf32>
        %42 = tensor.empty(%0) : tensor<?x1xf32>
        %43 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded_11 : tensor<?x1xf32>) outs(%42 : tensor<?x1xf32>) {
        ^bb0(%in: f32, %out: f32):
          %57 = arith.truncf %cst_0 : f64 to f32
          %58 = arith.mulf %in, %57 : f32
          linalg.yield %58 : f32
        } -> tensor<?x1xf32>
        %extracted_slice_12 = tensor.extract_slice %41[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_13 = tensor.expand_shape %extracted_slice_12 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %44 = tensor.empty(%2) : tensor<1x?xf32>
        %45 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%expanded_13 : tensor<1x?xf32>) outs(%44 : tensor<1x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %57 = arith.truncf %cst_0 : f64 to f32
          %58 = arith.mulf %in, %57 : f32
          linalg.yield %58 : f32
        } -> tensor<1x?xf32>
        %46 = tensor.empty(%0, %2) : tensor<?x?xf32>
        %47 = linalg.generic {indexing_maps = [#map3, #map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%43, %45 : tensor<?x1xf32>, tensor<1x?xf32>) outs(%46 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_18: f32, %out: f32):
          %57 = arith.subf %in, %in_18 : f32
          linalg.yield %57 : f32
        } -> tensor<?x?xf32>
        %48 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%47 : tensor<?x?xf32>) outs(%46 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %57 = math.powf %cst, %in : f32
          linalg.yield %57 : f32
        } -> tensor<?x?xf32>
        %49 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%38, %48 : tensor<?x?xf16>, tensor<?x?xf32>) outs(%46 : tensor<?x?xf32>) {
        ^bb0(%in: f16, %in_18: f32, %out: f32):
          %57 = arith.extf %in : f16 to f32
          %58 = arith.mulf %57, %in_18 : f32
          linalg.yield %58 : f32
        } -> tensor<?x?xf32>
        %subview_14 = memref.subview %arg2[%arg11, %7, %8, %37] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %50 = bufferization.to_tensor %subview_14 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %51 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%50 : tensor<?xf16>) outs(%40 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %57 = arith.extf %in : f16 to f32
          linalg.yield %57 : f32
        } -> tensor<?xf32>
        %extracted_slice_15 = tensor.extract_slice %51[0] [%2] [1] : tensor<?xf32> to tensor<?xf32>
        %expanded_16 = tensor.expand_shape %extracted_slice_15 [[0, 1]] output_shape [1, %2] : tensor<?xf32> into tensor<1x?xf32>
        %52 = linalg.generic {indexing_maps = [#map2, #map5, #map2], iterator_types = ["parallel", "parallel"]} ins(%49, %expanded_16 : tensor<?x?xf32>, tensor<1x?xf32>) outs(%46 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_18: f32, %out: f32):
          %57 = arith.mulf %in, %in_18 : f32
          linalg.yield %57 : f32
        } -> tensor<?x?xf32>
        %53 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %54 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%52 : tensor<?x?xf32>) outs(%53 : tensor<?x?xf16>) {
        ^bb0(%in: f32, %out: f16):
          %57 = arith.truncf %in : f32 to f16
          linalg.yield %57 : f16
        } -> tensor<?x?xf16>
        %subview_17 = memref.subview %arg3[%arg11, %15, %7, %18] [1, %2, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %55 = bufferization.to_tensor %subview_17 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
        %56 = linalg.matmul ins(%54, %55 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        scf.yield %56 : tensor<?x?xf32>
      }
      %subview_5 = memref.subview %arg6[%7] [1] [1] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      %27 = bufferization.to_tensor %subview_5 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %28 = tensor.empty() : tensor<f32>
      %29 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = []} ins(%27 : tensor<f16>) outs(%28 : tensor<f32>) {
      ^bb0(%in: f16, %out: f32):
        %37 = arith.extf %in : f16 to f32
        linalg.yield %37 : f32
      } -> tensor<f32>
      %subview_6 = memref.subview %arg3[%arg11, %15, %7, %18] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %30 = bufferization.to_tensor %subview_6 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
      %31 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%30 : tensor<?x?xf16>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %37 = arith.extf %in : f16 to f32
        linalg.yield %37 : f32
      } -> tensor<?x?xf32>
      %32 = linalg.generic {indexing_maps = [#map2, #map7, #map2], iterator_types = ["parallel", "parallel"]} ins(%31, %29 : tensor<?x?xf32>, tensor<f32>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %37 = arith.mulf %in, %in_8 : f32
        linalg.yield %37 : f32
      } -> tensor<?x?xf32>
      %33 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%26, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %37 = arith.addf %in, %in_8 : f32
        linalg.yield %37 : f32
      } -> tensor<?x?xf32>
      %34 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %35 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%33 : tensor<?x?xf32>) outs(%34 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %out: f16):
        %37 = arith.truncf %in : f32 to f16
        linalg.yield %37 : f16
      } -> tensor<?x?xf16>
      %subview_7 = memref.subview %arg7[%arg11, %15, %7, %18] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %36 = bufferization.to_buffer %35 : tensor<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      memref.copy %36, %subview_7 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    }
    return
  }
}