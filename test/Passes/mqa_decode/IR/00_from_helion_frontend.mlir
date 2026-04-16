#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
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
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0) : tensor<?x32xf16>
      %6 = linalg.fill ins(%cst_2 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %7 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %8 = tensor.empty(%0) : tensor<?x32x128xf16>
      %9 = linalg.fill ins(%cst_0 : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %10 = arith.muli %arg4, %0 : index
      %subview = memref.subview %arg3[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      %11 = bufferization.to_tensor %subview : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to tensor<?x32x128xf16>
      %12 = arith.muli %arg5, %1 : index
      %13 = arith.addi %12, %1 : index
      %14 = arith.subi %13, %12 : index
      %15 = arith.ceildivui %14, %2 : index
      %16:3 = scf.for %arg6 = %c0 to %15 step %c1 iter_args(%arg7 = %6, %arg8 = %7, %arg9 = %9) -> (tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>) {
        %20 = arith.muli %arg6, %2 : index
        %21 = arith.addi %12, %20 : index
        %subview_4 = memref.subview %arg0[%10, 0, %21] [%0, 128, %2] [1, 1, 1] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        %22 = bufferization.to_tensor %subview_4 : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to tensor<?x128x?xf16>
        %23 = arith.index_cast %0 : index to i64
        %24 = arith.cmpi eq, %23, %23 : i64
        cf.assert %24, "mismatching contracting dimension"
        %25 = tensor.empty(%0, %2) : tensor<?x32x?xf16>
        %26 = linalg.fill ins(%cst_0 : f16) outs(%25 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %27 = linalg.batch_matmul ins(%11, %22 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%26 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %28 = tensor.empty(%0) : tensor<?x32xi64>
        %29 = linalg.fill ins(%c0_i64 : i64) outs(%28 : tensor<?x32xi64>) -> tensor<?x32xi64>
        %30:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%27 : tensor<?x32x?xf16>) outs(%6, %29 : tensor<?x32xf16>, tensor<?x32xi64>) {
        ^bb0(%in: f16, %out: f16, %out_8: i64):
          %48 = linalg.index 2 : index
          %49 = arith.index_cast %48 : index to i64
          %50 = arith.maximumf %in, %out : f16
          %51 = arith.cmpf ogt, %in, %out : f16
          %52 = arith.select %51, %49, %out_8 : i64
          linalg.yield %50, %52 : f16, i64
        } -> (tensor<?x32xf16>, tensor<?x32xi64>)
        %31 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%30#0 : tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %48 = arith.mulf %in, %cst_3 : f16
          linalg.yield %48 : f16
        } -> tensor<?x32xf16>
        %32 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg7, %31 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %48 = arith.cmpf ogt, %in, %in_8 : f16
          %49 = arith.select %48, %in, %in_8 : f16
          linalg.yield %49 : f16
        } -> tensor<?x32xf16>
        %33 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27 : tensor<?x32x?xf16>) outs(%25 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %48 = arith.mulf %in, %cst_3 : f16
          linalg.yield %48 : f16
        } -> tensor<?x32x?xf16>
        %extracted_slice = tensor.extract_slice %32[0, 0] [%0, 32] [1, 1] : tensor<?x32xf16> to tensor<?x32xf16>
        %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%0, 32, 1] : tensor<?x32xf16> into tensor<?x32x1xf16>
        %34 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %expanded : tensor<?x32x?xf16>, tensor<?x32x1xf16>) outs(%25 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %48 = arith.subf %in, %in_8 : f16
          linalg.yield %48 : f16
        } -> tensor<?x32x?xf16>
        %35 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34 : tensor<?x32x?xf16>) outs(%25 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %48 = math.powf %cst, %in : f16
          linalg.yield %48 : f16
        } -> tensor<?x32x?xf16>
        %36 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %37 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%35 : tensor<?x32x?xf16>) outs(%36 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %48 = arith.addf %in, %out : f16
          linalg.yield %48 : f16
        } -> tensor<?x32xf16>
        %38 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg7, %32 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %48 = arith.subf %in, %in_8 : f16
          linalg.yield %48 : f16
        } -> tensor<?x32xf16>
        %39 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%38 : tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %48 = math.powf %cst, %in : f16
          linalg.yield %48 : f16
        } -> tensor<?x32xf16>
        %40 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg8, %39 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %48 = arith.mulf %in, %in_8 : f16
          linalg.yield %48 : f16
        } -> tensor<?x32xf16>
        %41 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%40, %37 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %48 = arith.addf %in, %in_8 : f16
          linalg.yield %48 : f16
        } -> tensor<?x32xf16>
        %extracted_slice_5 = tensor.extract_slice %39[0, 0] [%0, 32] [1, 1] : tensor<?x32xf16> to tensor<?x32xf16>
        %expanded_6 = tensor.expand_shape %extracted_slice_5 [[0], [1, 2]] output_shape [%0, 32, 1] : tensor<?x32xf16> into tensor<?x32x1xf16>
        %42 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %expanded_6 : tensor<?x32x128xf16>, tensor<?x32x1xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %48 = arith.mulf %in, %in_8 : f16
          linalg.yield %48 : f16
        } -> tensor<?x32x128xf16>
        %subview_7 = memref.subview %arg1[%10, %21, 0] [%0, %2, 128] [1, 1, 1] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        %43 = bufferization.to_tensor %subview_7 : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %24, "mismatching contracting dimension"
        %44 = arith.index_cast %2 : index to i64
        %45 = arith.cmpi eq, %44, %44 : i64
        cf.assert %45, "mismatching contracting dimension"
        %46 = linalg.batch_matmul ins(%35, %43 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %47 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %42 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_8: f16, %out: f16):
          %48 = arith.addf %in, %in_8 : f16
          linalg.yield %48 : f16
        } -> tensor<?x32x128xf16>
        scf.yield %32, %41, %47 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>
      }
      %17 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%16#1 : tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
      ^bb0(%in: f16, %out: f16):
        %20 = math.log2 %in : f16
        linalg.yield %20 : f16
      } -> tensor<?x32xf16>
      %18 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%17, %16#0 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %20 = arith.addf %in, %in_4 : f16
        linalg.yield %20 : f16
      } -> tensor<?x32xf16>
      %19 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %19 {
        %20 = tensor.empty(%4, %0) : tensor<?x?x32xf16>
        %21 = "loom.gather"(%18, %20, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x32xf16>, tensor<?x?x32xf16>, index) -> tensor<?x?x32xf16>
        %22 = tensor.empty(%0) : tensor<?x32xi64>
        %23 = linalg.fill ins(%c0_i64 : i64) outs(%22 : tensor<?x32xi64>) -> tensor<?x32xi64>
        %24:2 = linalg.generic {indexing_maps = [#map, #map4, #map4], iterator_types = ["reduction", "parallel", "parallel"]} ins(%21 : tensor<?x?x32xf16>) outs(%6, %23 : tensor<?x32xf16>, tensor<?x32xi64>) {
        ^bb0(%in: f16, %out: f16, %out_5: i64):
          %36 = linalg.index 0 : index
          %37 = arith.index_cast %36 : index to i64
          %38 = arith.maximumf %in, %out : f16
          %39 = arith.cmpf ogt, %in, %out : f16
          %40 = arith.select %39, %37, %out_5 : i64
          linalg.yield %38, %40 : f16, i64
        } -> (tensor<?x32xf16>, tensor<?x32xi64>)
        %25 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%21, %24#0 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%20 : tensor<?x?x32xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %36 = arith.subf %in, %in_5 : f16
          linalg.yield %36 : f16
        } -> tensor<?x?x32xf16>
        %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<?x?x32xf16>) outs(%20 : tensor<?x?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %36 = math.powf %cst, %in : f16
          linalg.yield %36 : f16
        } -> tensor<?x?x32xf16>
        %27 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %28 = linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["reduction", "parallel", "parallel"]} ins(%26 : tensor<?x?x32xf16>) outs(%27 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %36 = arith.addf %in, %out : f16
          linalg.yield %36 : f16
        } -> tensor<?x32xf16>
        %29 = linalg.generic {indexing_maps = [#map, #map4, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %28 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%20 : tensor<?x?x32xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %36 = arith.divf %in, %in_5 : f16
          linalg.yield %36 : f16
        } -> tensor<?x?x32xf16>
        %30 = tensor.empty(%4, %0) : tensor<?x?x32x128xf16>
        %31 = "loom.gather"(%16#2, %30, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x32x128xf16>, tensor<?x?x32x128xf16>, index) -> tensor<?x?x32x128xf16>
        %extracted_slice = tensor.extract_slice %29[0, 0, 0] [%4, %0, 32] [1, 1, 1] : tensor<?x?x32xf16> to tensor<?x?x32xf16>
        %expanded = tensor.expand_shape %extracted_slice [[0], [1], [2, 3]] output_shape [%4, %0, 32, 1] : tensor<?x?x32xf16> into tensor<?x?x32x1xf16>
        %32 = arith.cmpi eq, %4, %4 : index
        cf.assert %32, "mismatched size for broadcast"
        %33 = linalg.generic {indexing_maps = [#map5, #map6, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %expanded : tensor<?x?x32x128xf16>, tensor<?x?x32x1xf16>) outs(%30 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %36 = arith.mulf %in, %in_5 : f16
          linalg.yield %36 : f16
        } -> tensor<?x?x32x128xf16>
        %34 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%33 : tensor<?x?x32x128xf16>) outs(%9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %36 = arith.addf %in, %out : f16
          linalg.yield %36 : f16
        } -> tensor<?x32x128xf16>
        %subview_4 = memref.subview %arg2[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %35 = bufferization.to_buffer %34 : tensor<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        memref.copy %35, %subview_4 : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      }
    }
    return
  }
}