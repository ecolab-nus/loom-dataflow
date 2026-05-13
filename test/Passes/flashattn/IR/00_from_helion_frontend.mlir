#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @attention(%k_view_arg: memref<32x128x4096xf16>, %v_view_arg: memref<32x4096x128xf16>, %q_view_arg: memref<32x4096x128xf16>, %out__arg: memref<32x4096x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %0 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_b, upper_bound = 32 : index} : () -> index
    %1 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_m, upper_bound = 4096 : index} : () -> index
    %2 = "loom.sym"() {is_reduction = false, symbol_ref = @tile_n, upper_bound = 4096 : index} : () -> index
    %3 = arith.ceildivui %c32, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?x1xf16>
      %6 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %7 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %8 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
      %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %10 = arith.muli %arg4, %0 : index
      %11 = arith.muli %arg5, %1 : index
      %subview = memref.subview %q_view_arg[%10, %11, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %12 = bufferization.to_tensor %subview : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
      %13 = arith.ceildivui %c4096, %2 : index
      %14:3 = scf.for %arg6 = %c0 to %13 step %c1 iter_args(%arg7 = %6, %arg8 = %7, %arg9 = %9) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
        %18 = arith.muli %arg6, %2 : index
        %subview_4 = memref.subview %k_view_arg[%10, 0, %18] [%0, 128, %2] [1, 1, 1] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        %19 = bufferization.to_tensor %subview_4 : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf16>
        %20 = arith.index_cast %0 : index to i64
        %21 = arith.cmpi eq, %20, %20 : i64
        cf.assert %21, "mismatching contracting dimension"
        %22 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
        %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %24 = linalg.batch_matmul ins(%12, %19 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%23 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %25 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%24 : tensor<?x?x?xf16>) outs(%6 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.maximumf %in, %out : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x1xf16>
        %26 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.mulf %in, %cst_2 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x1xf16>
        %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %26 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %44 = arith.cmpf ogt, %in, %in_6 : f16
          %45 = arith.select %44, %in, %in_6 : f16
          linalg.yield %45 : f16
        } -> tensor<?x?x1xf16>
        %28 = "loom.broadcast"(%27, %22) {dim = 2 : i64} : (tensor<?x?x1xf16>, tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %29 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<?x?x?xf16>) outs(%22 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.mulf %in, %cst_2 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x?xf16>
        %30 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %28 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%22 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %44 = arith.subf %in, %in_6 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x?xf16>
        %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30 : tensor<?x?x?xf16>) outs(%22 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = math.exp %in : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x?xf16>
        %32 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %33 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%31 : tensor<?x?x?xf16>) outs(%32 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = arith.addf %in, %out : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x1xf16>
        %34 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %27 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %44 = arith.subf %in, %in_6 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x1xf16>
        %35 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34 : tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %44 = math.exp %in : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x1xf16>
        %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %35 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %44 = arith.mulf %in, %in_6 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x1xf16>
        %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %33 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %44 = arith.addf %in, %in_6 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x1xf16>
        %38 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %35 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %44 = arith.mulf %in, %in_6 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x128xf16>
        %subview_5 = memref.subview %v_view_arg[%10, %18, 0] [%0, %2, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %39 = bufferization.to_tensor %subview_5 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %21, "mismatching contracting dimension"
        %40 = arith.index_cast %2 : index to i64
        %41 = arith.cmpi eq, %40, %40 : i64
        cf.assert %41, "mismatching contracting dimension"
        %42 = linalg.batch_matmul ins(%31, %39 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %43 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %38 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %44 = arith.addf %in, %in_6 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?x128xf16>
        scf.yield %27, %37, %43 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
      }
      %15 = "loom.broadcast"(%14#1, %8) {dim = 2 : i64} : (tensor<?x?x1xf16>, tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#2, %15 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%8 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %18 = arith.divf %in, %in_4 : f16
        linalg.yield %18 : f16
      } -> tensor<?x?x128xf16>
      %subview_3 = memref.subview %out__arg[%10, %11, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %17 = bufferization.to_buffer %16 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      memref.copy %17, %subview_3 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
    }
    return
  }
}