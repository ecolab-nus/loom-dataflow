#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%k_view_arg: memref<16x128x8192xf16>, %v_view_arg: memref<16x8192x128xf16>, %q_view_arg: memref<16x32x128xf16>, %out__arg: memref<16x32x128xf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c16 = arith.constant 16 : index
    %c8192 = arith.constant 8192 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 16 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_s, upper_bound = 8192 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 8192 : index} : () -> index
    %3 = arith.ceildivui %c16, %0 : index
    %4 = arith.ceildivui %c8192, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0) : tensor<?x32x1xf16>
      %6 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %7 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %8 = tensor.empty(%0) : tensor<?x32x128xf16>
      %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %10 = arith.muli %arg4, %0 : index
      %subview = memref.subview %q_view_arg[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      %11 = bufferization.to_tensor %subview : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to tensor<?x32x128xf16>
      %12 = arith.muli %arg5, %1 : index
      %13 = arith.addi %12, %1 : index
      %14 = arith.subi %13, %12 : index
      %15 = arith.ceildivui %14, %2 : index
      %16:3 = scf.for %arg6 = %c0 to %15 step %c1 iter_args(%arg7 = %6, %arg8 = %7, %arg9 = %9) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %28 = arith.muli %arg6, %2 : index
        %29 = arith.addi %12, %28 : index
        %subview_3 = memref.subview %k_view_arg[%10, 0, %29] [%0, 128, %2] [1, 1, 1] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        %30 = bufferization.to_tensor %subview_3 : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to tensor<?x128x?xf16>
        %31 = arith.index_cast %0 : index to i64
        %32 = arith.cmpi eq, %31, %31 : i64
        cf.assert %32, "mismatching contracting dimension"
        %33 = tensor.empty(%0, %2) : tensor<?x32x?xf16>
        %34 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %35 = linalg.batch_matmul ins(%11, %30 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%34 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %36 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%35 : tensor<?x32x?xf16>) outs(%6 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %56 = arith.maximumf %in, %out : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x1xf16>
        %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %56 = arith.mulf %in, %cst_2 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x1xf16>
        %38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %37 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.cmpf ogt, %in, %in_5 : f16
          %57 = arith.select %56, %in, %in_5 : f16
          linalg.yield %57 : f16
        } -> tensor<?x32x1xf16>
        %39 = "loom.broadcast"(%38, %33) {dim = 2 : i64} : (tensor<?x32x1xf16>, tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %40 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35 : tensor<?x32x?xf16>) outs(%33 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %56 = arith.mulf %in, %cst_2 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x?xf16>
        %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %39 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%33 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.subf %in, %in_5 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x?xf16>
        %42 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41 : tensor<?x32x?xf16>) outs(%33 : tensor<?x32x?xf16>) {
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
        %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %38 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
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
        %49 = "loom.broadcast"(%46, %8) {dim = 2 : i64} : (tensor<?x32x1xf16>, tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %50 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %49 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.mulf %in, %in_5 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x128xf16>
        %subview_4 = memref.subview %v_view_arg[%10, %29, 0] [%0, %2, 128] [1, 1, 1] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        %51 = bufferization.to_tensor %subview_4 : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %32, "mismatching contracting dimension"
        %52 = arith.index_cast %2 : index to i64
        %53 = arith.cmpi eq, %52, %52 : i64
        cf.assert %53, "mismatching contracting dimension"
        %54 = linalg.batch_matmul ins(%42, %51 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %55 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54, %50 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %56 = arith.addf %in, %in_5 : f16
          linalg.yield %56 : f16
        } -> tensor<?x32x128xf16>
        scf.yield %38, %48, %55 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      %17 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16#1 : tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %out: f16):
        %28 = math.log %in : f16
        linalg.yield %28 : f16
      } -> tensor<?x32x1xf16>
      %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%17, %16#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %28 = arith.addf %in, %in_3 : f16
        linalg.yield %28 : f16
      } -> tensor<?x32x1xf16>
      %19 = "loom.broadcast"(%16#1, %8) {dim = 2 : i64} : (tensor<?x32x1xf16>, tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16#2, %19 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%8 : tensor<?x32x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %28 = arith.divf %in, %in_3 : f16
        linalg.yield %28 : f16
      } -> tensor<?x32x128xf16>
      %21 = bufferization.to_buffer %18 : tensor<?x32x1xf16> to memref<?x32x1xf16>
      %22 = "loom.placeholder"(%4, %0) {static_sizes = array<i64: -9223372036854775808, -9223372036854775808, 32, 1>} : (index, index) -> memref<?x?x32x1xf16>
      "loom.gather"(%21, %22, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0>, static_area = array<i64: 1, 1>} : (memref<?x32x1xf16>, memref<?x?x32x1xf16>, index) -> ()
      %23 = bufferization.to_tensor %22 : memref<?x?x32x1xf16> to tensor<?x?x32x1xf16>
      %24 = bufferization.to_buffer %20 : tensor<?x32x128xf16> to memref<?x32x128xf16>
      %25 = "loom.placeholder"(%4, %0) {static_sizes = array<i64: -9223372036854775808, -9223372036854775808, 32, 128>} : (index, index) -> memref<?x?x32x128xf16>
      "loom.gather"(%24, %25, %arg5) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0>, static_area = array<i64: 1, 1>} : (memref<?x32x128xf16>, memref<?x?x32x128xf16>, index) -> ()
      %26 = bufferization.to_tensor %25 : memref<?x?x32x128xf16> to tensor<?x?x32x128xf16>
      %27 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %27 {
        %28 = tensor.empty(%0) : tensor<?x32x1xi64>
        %29 = linalg.fill ins(%c0_i64 : i64) outs(%28 : tensor<?x32x1xi64>) -> tensor<?x32x1xi64>
        %30:2 = linalg.generic {indexing_maps = [#map2, #map3, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%23 : tensor<?x?x32x1xf16>) outs(%6, %29 : tensor<?x32x1xf16>, tensor<?x32x1xi64>) {
        ^bb0(%in: f16, %out: f16, %out_4: i64):
          %43 = linalg.index 0 : index
          %44 = arith.index_cast %43 : index to i64
          %45 = arith.maximumf %in, %out : f16
          %46 = arith.cmpf ogt, %in, %out : f16
          %47 = arith.select %46, %44, %out_4 : i64
          linalg.yield %45, %47 : f16, i64
        } -> (tensor<?x32x1xf16>, tensor<?x32x1xi64>)
        %31 = tensor.empty(%4, %0) : tensor<?x?x32x1xf16>
        %32 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %30#0 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%31 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %43 = arith.subf %in, %in_4 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x32x1xf16>
        %33 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<?x?x32x1xf16>) outs(%31 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %43 = math.exp %in : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x32x1xf16>
        %34 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %35 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%33 : tensor<?x?x32x1xf16>) outs(%34 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %43 = arith.addf %in, %out : f16
          linalg.yield %43 : f16
        } -> tensor<?x32x1xf16>
        %36 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33, %35 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%31 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %43 = arith.divf %in, %in_4 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x32x1xf16>
        %37 = tensor.empty(%4, %0) : tensor<?x?x32x128xf16>
        %38 = "loom.broadcast"(%36, %37) {dim = 3 : i64} : (tensor<?x?x32x1xf16>, tensor<?x?x32x128xf16>) -> tensor<?x?x32x128xf16>
        %39 = arith.cmpi eq, %4, %1 : index
        cf.assert %39, "mismatched size for broadcast"
        %40 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %38 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%37 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %43 = arith.mulf %in, %in_4 : f16
          linalg.yield %43 : f16
        } -> tensor<?x?x32x128xf16>
        %41 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%40 : tensor<?x?x32x128xf16>) outs(%9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %43 = arith.addf %in, %out : f16
          linalg.yield %43 : f16
        } -> tensor<?x32x128xf16>
        %subview_3 = memref.subview %out__arg[%10, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %42 = bufferization.to_buffer %41 : tensor<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        memref.copy %42, %subview_3 : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      }
    }
    return
  }
}