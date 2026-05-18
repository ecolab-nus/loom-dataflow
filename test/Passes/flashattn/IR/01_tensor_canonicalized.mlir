module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @attention(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %0 = loom.sym @tile_b {upper_bound = 32 : index} : index
    %1 = loom.sym @tile_m {upper_bound = 4096 : index} : index
    %2 = loom.sym @tile_n {upper_bound = 4096 : index} : index
    %3 = arith.ceildivui %c32, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?x1xf16>
      %6 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
      %7 = arith.muli %arg4, %0 : index
      %8 = arith.muli %arg5, %1 : index
      %subview = memref.subview %arg2[%7, %8, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %9 = loom.bufferize_to_tensor %subview[%0, %1, 128] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> -> tensor<?x?x128xf16>
      %10 = arith.ceildivui %c4096, %2 : index
      %11 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %12 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %13 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %14:3 = scf.for %arg6 = %c0 to %10 step %c1 iter_args(%arg7 = %13, %arg8 = %12, %arg9 = %11) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
        %18 = arith.muli %arg6, %2 : index
        %subview_4 = memref.subview %arg0[%7, 0, %18] [%0, 128, %2] [1, 1, 1] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        %19 = loom.bufferize_to_tensor %subview_4[%0, 128, %2] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> -> tensor<?x128x?xf16>
        %20 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
        %21 = linalg.fill ins(%cst : f16) outs(%20 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %22 = linalg.batch_matmul ins(%9, %19 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%21 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %23 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%22 : tensor<?x?x?xf16>) outs(%23 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = arith.maximumf %in, %out : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x1xf16>
        %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%24 : tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = arith.mulf %in, %cst_2 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x1xf16>
        %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %25 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %38 = arith.cmpf ogt, %in, %in_6 : f16
          %39 = arith.select %38, %in, %in_6 : f16
          linalg.yield %39 : f16
        } -> tensor<?x?x1xf16>
        %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<?x?x?xf16>) outs(%20 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = arith.mulf %in, %cst_2 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x?xf16>
        %28 = loom.broadcast ins(%26 : tensor<?x?x1xf16>) outs(%20 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
        %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %28 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%20 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %38 = arith.subf %in, %in_6 : f16
          %39 = math.exp %38 : f16
          linalg.yield %39 : f16
        } -> tensor<?x?x?xf16>
        %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %26 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %38 = arith.subf %in, %in_6 : f16
          %39 = math.exp %38 : f16
          linalg.yield %39 : f16
        } -> tensor<?x?x1xf16>
        %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %30 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %38 = arith.mulf %in, %in_6 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x1xf16>
        %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%29 : tensor<?x?x?xf16>) outs(%31 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %38 = arith.addf %in, %out : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x1xf16>
        %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %30 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%6 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %38 = arith.mulf %in, %in_6 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x128xf16>
        %subview_5 = memref.subview %arg1[%7, %18, 0] [%0, %2, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %34 = loom.bufferize_to_tensor %subview_5[%0, %2, 128] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> -> tensor<?x?x128xf16>
        %35 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %36 = linalg.batch_matmul ins(%29, %34 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%35 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %33 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%6 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %38 = arith.addf %in, %in_6 : f16
          linalg.yield %38 : f16
        } -> tensor<?x?x128xf16>
        scf.yield %26, %32, %37 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
      }
      %15 = loom.broadcast ins(%14#1 : tensor<?x?x1xf16>) outs(%6 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
      %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#2, %15 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%6 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %18 = arith.divf %in, %in_4 : f16
        linalg.yield %18 : f16
      } -> tensor<?x?x128xf16>
      %subview_3 = memref.subview %arg3[%7, %8, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %17 = loom.bufferize_to_memref %16 : tensor<?x?x128xf16> -> memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      memref.copy %17, %subview_3 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
    }
    return
  }
}
