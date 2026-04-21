#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c8192 = arith.constant 8192 : index
    %c16 = arith.constant 16 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 16 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_s, upper_bound = 8192 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 64 : index} : () -> index
    %3 = arith.ceildivui %c16, %0 : index
    %4 = arith.ceildivui %c8192, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0) : tensor<?x32x1xf16>
      %6 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %7 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %8 = tensor.empty(%0) : tensor<?x32x128xf16>
      %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %10 = arith.muli %arg4, %0 : index
      %subview = memref.subview %arg3[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      %11 = bufferization.to_tensor %subview : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to tensor<?x32x128xf16>
      %12 = arith.muli %arg5, %1 : index
      %13 = arith.addi %12, %1 : index
      %14 = arith.subi %13, %12 : index
      %15 = arith.ceildivui %14, %2 : index
      %16:3 = scf.for %arg6 = %c0 to %15 step %c1 iter_args(%arg7 = %6, %arg8 = %7, %arg9 = %9) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %27 = arith.muli %arg6, %2 : index
        %28 = arith.addi %12, %27 : index
        %subview_3 = memref.subview %arg0[%10, 0, %28] [%0, 128, %2] [1, 1, 1] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        %29 = bufferization.to_tensor %subview_3 : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to tensor<?x128x?xf16>
        %30 = arith.index_cast %0 : index to i64
        %31 = arith.cmpi eq, %30, %30 : i64
        cf.assert %31, "mismatching contracting dimension"
        %32 = tensor.empty(%0, %2) : tensor<?x32x?xf16>
        %33 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %34 = linalg.batch_matmul ins(%11, %29 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%33 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %35 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%34 : tensor<?x32x?xf16>) outs(%6 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %56 = arith.maximumf %in, %out : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x1xf16>
        %36 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %56 = arith.mulf %in, %cst_2 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x1xf16>
        %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %36 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.cmpf ogt, %in, %in_5 : f16
          %57 = arith.select %56, %in, %in_5 : f16
          linalg.yield %57 : f16
        } -> tensor<?x32x1xf16>
        %38 = tensor.empty(%0) : tensor<?x32x32xf16>
        %39 = "loom.broadcast"(%37, %38) {dim = 2 : i64} : (tensor<?x32x1xf16>, tensor<?x32x32xf16>) -> tensor<?x32x?xf16>
        %40 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34 : tensor<?x32x?xf16>) outs(%32 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %56 = arith.mulf %in, %cst_2 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x?xf16>
        %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %39 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%32 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.subf %in, %in_5 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x?xf16>
        %42 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41 : tensor<?x32x?xf16>) outs(%32 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %56 = math.exp %in : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x?xf16>
        %43 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %44 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%42 : tensor<?x32x?xf16>) outs(%43 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %56 = arith.addf %in, %out : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x1xf16>
        %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %37 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.subf %in, %in_5 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x1xf16>
        %46 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %56 = math.exp %in : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x1xf16>
        %47 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %46 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.mulf %in, %in_5 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x1xf16>
        %48 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %44 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.addf %in, %in_5 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x1xf16>
        %49 = "loom.broadcast"(%46, %38) {dim = 2 : i64} : (tensor<?x32x1xf16>, tensor<?x32x32xf16>) -> tensor<?x32x128xf16>
        %50 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %49 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.mulf %in, %in_5 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x128xf16>
        %subview_4 = memref.subview %arg1[%10, %28, 0] [%0, %2, 128] [1, 1, 1] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        %51 = bufferization.to_tensor %subview_4 : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %31, "mismatching contracting dimension"
        %52 = arith.index_cast %2 : index to i64
        %53 = arith.cmpi eq, %52, %52 : i64
        cf.assert %53, "mismatching contracting dimension"
        %54 = linalg.batch_matmul ins(%42, %51 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %55 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54, %50 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.addf %in, %in_5 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x128xf16>
        scf.yield %37, %48, %55 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16#1 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %out: f16):
        %27 = math.log %in : f16
        linalg.yield %27 : f16
      } -> tensor<?x32x1xf16>
      %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %16#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %27 = arith.addf %in, %in_3 : f16
        linalg.yield %27 : f16
      } -> tensor<?x32x1xf16>
      %19 = tensor.empty(%0) : tensor<?x32x32xf16>
      %20 = "loom.broadcast"(%16#1, %19) {dim = 2 : i64} : (tensor<?x32x1xf16>, tensor<?x32x32xf16>) -> tensor<?x32x128xf16>
      %21 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16#2, %20 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %27 = arith.divf %in, %in_3 : f16
        linalg.yield %27 : f16
      } -> tensor<?x32x128xf16>
      %22 = tensor.empty(%4, %0) : tensor<?x?x32x1xf16>
      %23 = "loom.gather"(%18, %22, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x32x1xf16>, tensor<?x?x32x1xf16>, index) -> tensor<?x?x32x1xf16>
      %24 = tensor.empty(%4, %0) : tensor<?x?x32x128xf16>
      %25 = "loom.gather"(%21, %24, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x32x128xf16>, tensor<?x?x32x128xf16>, index) -> tensor<?x?x32x128xf16>
      %26 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %26 {
        %27 = tensor.empty(%0) : tensor<?x32x1xi64>
        %28 = linalg.fill ins(%c0_i64 : i64) outs(%27 : tensor<?x32x1xi64>) -> tensor<?x32x1xi64>
        %29:2 = linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%23 : tensor<?x?x32x1xf16>) outs(%6, %28 : tensor<?x32x1xf16>, tensor<?x32x1xi64>) {
        ^bb0(%in: f16, %out: f16, %out_4: i64):
          %41 = linalg.index 0 : index
          %42 = arith.index_cast %41 : index to i64
          %43 = arith.maximumf %in, %out : f16
          %44 = arith.cmpf ogt, %in, %out : f16
          %45 = arith.select %44, %42, %out_4 : i64
          linalg.yield %43, %45 : f16, i64
        } -> (tensor<?x32x1xf16>, tensor<?x32x1xi64>)
        %30 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %29#0 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%22 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %41 = arith.subf %in, %in_4 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?x32x1xf16>
        %31 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30 : tensor<?x?x32x1xf16>) outs(%22 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %41 = math.exp %in : f16
          linalg.yield %41 : f16
        } -> tensor<?x?x32x1xf16>
        %32 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %33 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%31 : tensor<?x?x32x1xf16>) outs(%32 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %41 = arith.addf %in, %out : f16
          linalg.yield %41 : f16
        } -> tensor<?x32x1xf16>
        %34 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%31, %33 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%22 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %41 = arith.divf %in, %in_4 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?x32x1xf16>
        %35 = tensor.empty(%4, %0) : tensor<?x?x32x32xf16>
        %36 = "loom.broadcast"(%34, %35) {dim = 3 : i64} : (tensor<?x?x32x1xf16>, tensor<?x?x32x32xf16>) -> tensor<?x?x32x128xf16>
        %37 = arith.cmpi eq, %4, %1 : index
        cf.assert %37, "mismatched size for broadcast"
        %38 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%25, %36 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%24 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %41 = arith.mulf %in, %in_4 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?x32x128xf16>
        %39 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%38 : tensor<?x?x32x128xf16>) outs(%9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %41 = arith.addf %in, %out : f16
          linalg.yield %41 : f16
        } -> tensor<?x32x128xf16>
        %subview_3 = memref.subview %arg2[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %40 = bufferization.to_buffer %39 : tensor<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        memref.copy %40, %subview_3 : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      }
    }
    return
  }
}