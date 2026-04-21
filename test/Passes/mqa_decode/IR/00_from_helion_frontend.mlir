#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
    %c3 = arith.constant 3 : index
    %c0_i64 = arith.constant 0 : i64
    %c2 = arith.constant 2 : index
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
        %24 = arith.muli %arg6, %2 : index
        %25 = arith.addi %12, %24 : index
        %subview_3 = memref.subview %arg0[%10, 0, %25] [%0, 128, %2] [1, 1, 1] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        %26 = bufferization.to_tensor %subview_3 : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to tensor<?x128x?xf16>
        %27 = arith.index_cast %0 : index to i64
        %28 = arith.cmpi eq, %27, %27 : i64
        cf.assert %28, "mismatching contracting dimension"
        %29 = tensor.empty(%0, %2) : tensor<?x32x?xf16>
        %30 = linalg.fill ins(%cst : f16) outs(%29 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %31 = linalg.batch_matmul ins(%11, %26 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%30 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %32 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%31 : tensor<?x32x?xf16>) outs(%6 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.maximumf %in, %out : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %33 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.mulf %in, %cst_2 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %34 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %33 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.cmpf ogt, %in, %in_5 : f16
          %54 = arith.select %53, %in, %in_5 : f16
          linalg.yield %54 : f16
        } -> tensor<?x32x1xf16>
        %35 = tensor.empty(%0) : tensor<?x32x32xf16>
        %36 = "loom.broadcast"(%34, %35, %c2) : (tensor<?x32x1xf16>, tensor<?x32x32xf16>, index) -> tensor<?x32x?xf16>
        %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31 : tensor<?x32x?xf16>) outs(%29 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.mulf %in, %cst_2 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x?xf16>
        %38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %36 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%29 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.subf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x?xf16>
        %39 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38 : tensor<?x32x?xf16>) outs(%29 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = math.exp %in : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x?xf16>
        %40 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %41 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%39 : tensor<?x32x?xf16>) outs(%40 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = arith.addf %in, %out : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %42 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %34 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.subf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %43 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %53 = math.exp %in : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %44 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %43 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.mulf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %41 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.addf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x1xf16>
        %46 = "loom.broadcast"(%43, %35, %c2) : (tensor<?x32x1xf16>, tensor<?x32x32xf16>, index) -> tensor<?x32x128xf16>
        %47 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %46 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.mulf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x128xf16>
        %subview_4 = memref.subview %arg1[%10, %25, 0] [%0, %2, 128] [1, 1, 1] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        %48 = bufferization.to_tensor %subview_4 : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %28, "mismatching contracting dimension"
        %49 = arith.index_cast %2 : index to i64
        %50 = arith.cmpi eq, %49, %49 : i64
        cf.assert %50, "mismatching contracting dimension"
        %51 = linalg.batch_matmul ins(%39, %48 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %52 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%51, %47 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %53 = arith.addf %in, %in_5 : f16
          linalg.yield %53 : f16
        } -> tensor<?x32x128xf16>
        scf.yield %34, %45, %52 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16#1 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %out: f16):
        %24 = math.log %in : f16
        linalg.yield %24 : f16
      } -> tensor<?x32x1xf16>
      %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %16#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %24 = arith.addf %in, %in_3 : f16
        linalg.yield %24 : f16
      } -> tensor<?x32x1xf16>
      %19 = tensor.empty(%4, %0) : tensor<?x?x32x1xf16>
      %20 = "loom.gather"(%18, %19, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x32x1xf16>, tensor<?x?x32x1xf16>, index) -> tensor<?x?x32x1xf16>
      %21 = tensor.empty(%4, %0) : tensor<?x?x32x128xf16>
      %22 = "loom.gather"(%16#2, %21, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x32x128xf16>, tensor<?x?x32x128xf16>, index) -> tensor<?x?x32x128xf16>
      %23 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %23 {
        %24 = tensor.empty(%0) : tensor<?x32x1xi64>
        %25 = linalg.fill ins(%c0_i64 : i64) outs(%24 : tensor<?x32x1xi64>) -> tensor<?x32x1xi64>
        %26:2 = linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%20 : tensor<?x?x32x1xf16>) outs(%6, %25 : tensor<?x32x1xf16>, tensor<?x32x1xi64>) {
        ^bb0(%in: f16, %out: f16, %out_4: i64):
          %38 = linalg.index 0 : index
          %39 = arith.index_cast %38 : index to i64
          %40 = arith.maximumf %in, %out : f16
          %41 = arith.cmpf ogt, %in, %out : f16
          %42 = arith.select %41, %39, %out_4 : i64
          linalg.yield %40, %42 : f16, i64
        } -> (tensor<?x32x1xf16>, tensor<?x32x1xi64>)
        %27 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %26#0 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%19 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %38 = arith.subf %in, %in_4 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x32x1xf16>
        %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%27 : tensor<?x?x32x1xf16>) outs(%19 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = math.exp %in : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x32x1xf16>
        %29 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %30 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%28 : tensor<?x?x32x1xf16>) outs(%29 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = arith.addf %in, %out : f16
          linalg.yield %38 : f16
        } -> tensor<?x32x1xf16>
        %31 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %30 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%19 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %38 = arith.divf %in, %in_4 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x32x1xf16>
        %32 = tensor.empty(%1, %0) : tensor<?x?x32x32xf16>
        %33 = "loom.broadcast"(%31, %32, %c3) : (tensor<?x?x32x1xf16>, tensor<?x?x32x32xf16>, index) -> tensor<?x?x32x128xf16>
        %34 = arith.cmpi eq, %4, %1 : index
        cf.assert %34, "mismatched size for broadcast"
        %35 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%22, %33 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%21 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %38 = arith.mulf %in, %in_4 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x32x128xf16>
        %36 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%35 : tensor<?x?x32x128xf16>) outs(%9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = arith.addf %in, %out : f16
          linalg.yield %38 : f16
        } -> tensor<?x32x128xf16>
        %subview_3 = memref.subview %arg2[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %37 = bufferization.to_buffer %36 : tensor<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        memref.copy %37, %subview_3 : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      }
    }
    return
  }
}