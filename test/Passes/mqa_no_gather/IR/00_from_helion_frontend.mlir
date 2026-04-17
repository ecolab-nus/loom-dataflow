#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>) {
    %cst = arith.constant 2.000000e+00 : f16
    %c0_i64 = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %cst_2 = arith.constant 0xFC00 : f16
    %cst_3 = arith.constant 1.275630e-01 : f16
    %c8192 = arith.constant 8192 : index
    %c16 = arith.constant 16 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 16 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_s, upper_bound = 8192 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 64 : index} : () -> index
    %3 = arith.ceildivui %c16, %0 : index
    %4 = arith.ceildivui %c8192, %1 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0) : tensor<?x32xf16>
      %6 = linalg.fill ins(%cst_2 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %7 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %8 = tensor.empty(%0) : tensor<?x32x128xf16>
      %9 = linalg.fill ins(%cst_0 : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %10 = arith.muli %arg3, %0 : index
      %subview = memref.subview %arg2[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      %11 = bufferization.to_tensor %subview : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to tensor<?x32x128xf16>
      %12 = arith.muli %arg4, %1 : index
      %13 = arith.addi %12, %1 : index
      %14 = arith.subi %13, %12 : index
      %15 = arith.ceildivui %14, %2 : index
      %16:3 = scf.for %arg5 = %c0 to %15 step %c1 iter_args(%arg6 = %6, %arg7 = %7, %arg8 = %9) -> (tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>) {
        %17 = arith.muli %arg5, %2 : index
        %18 = arith.addi %12, %17 : index
        %subview_4 = memref.subview %arg0[%10, 0, %18] [%0, 128, %2] [1, 1, 1] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        %19 = bufferization.to_tensor %subview_4 : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to tensor<?x128x?xf16>
        %20 = arith.index_cast %0 : index to i64
        %21 = arith.cmpi eq, %20, %20 : i64
        cf.assert %21, "mismatching contracting dimension"
        %22 = tensor.empty(%0, %2) : tensor<?x32x?xf16>
        %23 = linalg.fill ins(%cst_0 : f16) outs(%22 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %24 = linalg.batch_matmul ins(%11, %19 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%23 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %25 = tensor.empty(%0) : tensor<?x32xi64>
        %26 = linalg.fill ins(%c0_i64 : i64) outs(%25 : tensor<?x32xi64>) -> tensor<?x32xi64>
        %27:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%24 : tensor<?x32x?xf16>) outs(%6, %26 : tensor<?x32xf16>, tensor<?x32xi64>) {
        ^bb0(%in: f16, %out: f16, %out_8: i64):
          %45 = linalg.index 2 : index
          %46 = arith.index_cast %45 : index to i64
          %47 = arith.maximumf %in, %out : f16
          %48 = arith.cmpf ogt, %in, %out : f16
          %49 = arith.select %48, %46, %out_8 : i64
          linalg.yield %47, %49 : f16, i64
        } -> (tensor<?x32xf16>, tensor<?x32xi64>)
        %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%27#0 : tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %45 = arith.mulf %in, %cst_3 : f16
          linalg.yield %45 : f16
        } -> tensor<?x32xf16>
        %29 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg6, %28 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %45 = arith.cmpf ogt, %in, %in_8 : f16
          %46 = arith.select %45, %in, %in_8 : f16
          linalg.yield %46 : f16
        } -> tensor<?x32xf16>
        %30 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<?x32x?xf16>) outs(%22 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %45 = arith.mulf %in, %cst_3 : f16
          linalg.yield %45 : f16
        } -> tensor<?x32x?xf16>
        %extracted_slice = tensor.extract_slice %29[0, 0] [%0, 32] [1, 1] : tensor<?x32xf16> to tensor<?x32xf16>
        %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%0, 32, 1] : tensor<?x32xf16> into tensor<?x32x1xf16>
        %31 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %expanded : tensor<?x32x?xf16>, tensor<?x32x1xf16>) outs(%22 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %45 = arith.subf %in, %in_8 : f16
          linalg.yield %45 : f16
        } -> tensor<?x32x?xf16>
        %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31 : tensor<?x32x?xf16>) outs(%22 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %45 = math.powf %cst, %in : f16
          linalg.yield %45 : f16
        } -> tensor<?x32x?xf16>
        %33 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %34 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%32 : tensor<?x32x?xf16>) outs(%33 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %45 = arith.addf %in, %out : f16
          linalg.yield %45 : f16
        } -> tensor<?x32xf16>
        %35 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg6, %29 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %45 = arith.subf %in, %in_8 : f16
          linalg.yield %45 : f16
        } -> tensor<?x32xf16>
        %36 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%35 : tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %45 = math.powf %cst, %in : f16
          linalg.yield %45 : f16
        } -> tensor<?x32xf16>
        %37 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg7, %36 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %45 = arith.mulf %in, %in_8 : f16
          linalg.yield %45 : f16
        } -> tensor<?x32xf16>
        %38 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%37, %34 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %45 = arith.addf %in, %in_8 : f16
          linalg.yield %45 : f16
        } -> tensor<?x32xf16>
        %extracted_slice_5 = tensor.extract_slice %36[0, 0] [%0, 32] [1, 1] : tensor<?x32xf16> to tensor<?x32xf16>
        %expanded_6 = tensor.expand_shape %extracted_slice_5 [[0], [1, 2]] output_shape [%0, 32, 1] : tensor<?x32xf16> into tensor<?x32x1xf16>
        %39 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %expanded_6 : tensor<?x32x128xf16>, tensor<?x32x1xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %45 = arith.mulf %in, %in_8 : f16
          linalg.yield %45 : f16
        } -> tensor<?x32x128xf16>
        %subview_7 = memref.subview %arg1[%10, %18, 0] [%0, %2, 128] [1, 1, 1] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        %40 = bufferization.to_tensor %subview_7 : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %21, "mismatching contracting dimension"
        %41 = arith.index_cast %2 : index to i64
        %42 = arith.cmpi eq, %41, %41 : i64
        cf.assert %42, "mismatching contracting dimension"
        %43 = linalg.batch_matmul ins(%32, %40 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %44 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %39 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %45 = arith.addf %in, %in_8 : f16
          linalg.yield %45 : f16
        } -> tensor<?x32x128xf16>
        scf.yield %29, %38, %44 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>
      }
    }
    return
  }
}