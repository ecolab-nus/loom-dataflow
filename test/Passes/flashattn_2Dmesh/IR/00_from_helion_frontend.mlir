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
        %19 = arith.muli %arg6, %2 : index
        %subview_4 = memref.subview %k_view_arg[%10, 0, %19] [%0, 128, %2] [1, 1, 1] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        %20 = bufferization.to_tensor %subview_4 : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf16>
        %21 = arith.index_cast %0 : index to i64
        %22 = arith.cmpi eq, %21, %21 : i64
        cf.assert %22, "mismatching contracting dimension"
        %23 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
        %24 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %25 = linalg.batch_matmul ins(%12, %20 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%24 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %26 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%25 : tensor<?x?x?xf16>) outs(%6 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.maximumf %in, %out : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26 : tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.mulf %in, %cst_2 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %28 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %27 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.cmpf ogt, %in, %in_6 : f16
          %47 = arith.select %46, %in, %in_6 : f16
          linalg.yield %47 : f16
        } -> tensor<?x?x1xf16>
        %29 = tensor.empty(%0, %1) : tensor<?x?x32xf16>
        %30 = "loom.broadcast"(%28, %29) {dim = 2 : i64} : (tensor<?x?x1xf16>, tensor<?x?x32xf16>) -> tensor<?x?x?xf16>
        %31 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<?x?x?xf16>) outs(%23 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.mulf %in, %cst_2 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x?xf16>
        %32 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %30 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%23 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.subf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x?xf16>
        %33 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%32 : tensor<?x?x?xf16>) outs(%23 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = math.exp %in : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x?xf16>
        %34 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %35 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%33 : tensor<?x?x?xf16>) outs(%34 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.addf %in, %out : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %28 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.subf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36 : tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = math.exp %in : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %37 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.mulf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %39 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %35 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.addf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %40 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %37 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.mulf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x128xf16>
        %subview_5 = memref.subview %v_view_arg[%10, %19, 0] [%0, %2, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %41 = bufferization.to_tensor %subview_5 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        cf.assert %22, "mismatching contracting dimension"
        %42 = arith.index_cast %2 : index to i64
        %43 = arith.cmpi eq, %42, %42 : i64
        cf.assert %43, "mismatching contracting dimension"
        %44 = linalg.batch_matmul ins(%33, %41 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%9 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %45 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %40 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%8 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.addf %in, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x128xf16>
        scf.yield %28, %39, %45 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
      }
      %15 = tensor.empty(%0, %1) : tensor<?x?x32xf16>
      %16 = "loom.broadcast"(%14#1, %15) {dim = 2 : i64} : (tensor<?x?x1xf16>, tensor<?x?x32xf16>) -> tensor<?x?x128xf16>
      %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#2, %16 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%8 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %19 = arith.divf %in, %in_4 : f16
        linalg.yield %19 : f16
      } -> tensor<?x?x128xf16>
      %subview_3 = memref.subview %out__arg[%10, %11, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %18 = bufferization.to_buffer %17 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      memref.copy %18, %subview_3 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
    }
    return
  }
}