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
      %9 = bufferization.to_tensor %subview : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
      %10 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
      %11 = loom.sync ins(%9 : tensor<?x?x128xf16>) outs(%10 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %12 = arith.ceildivui %c4096, %2 : index
      %13 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %14 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %15 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %16:3 = scf.for %arg6 = %c0 to %12 step %c1 iter_args(%arg7 = %15, %arg8 = %14, %arg9 = %13) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
        %23 = arith.muli %arg6, %2 : index
        %subview_4 = memref.subview %arg0[%7, 0, %23] [%0, 128, %2] [1, 1, 1] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        %24 = bufferization.to_tensor %subview_4 : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf16>
        %25 = tensor.empty(%0, %2) : tensor<?x128x?xf16>
        %26 = loom.sync ins(%24 : tensor<?x128x?xf16>) outs(%25 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
        %27 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
        %28 = linalg.fill ins(%cst : f16) outs(%27 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %29 = linalg.batch_matmul ins(%11, %26 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%28 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %30 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%29 : tensor<?x?x?xf16>) outs(%30 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.maximumf %in, %out : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %31 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.mulf %in_6, %cst_2 : f16
          %47 = arith.cmpf ogt, %in, %46 : f16
          %48 = arith.select %47, %in, %46 : f16
          linalg.yield %48 : f16
        } -> tensor<?x?x1xf16>
        %33 = tensor.empty(%0, %1) : tensor<?x?x32xf16>
        %34 = loom.broadcast ins(%32 : tensor<?x?x1xf16>) outs(%33 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
        %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %34 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%27 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.mulf %in, %cst_2 : f16
          %47 = arith.subf %46, %in_6 : f16
          %48 = math.exp %47 : f16
          linalg.yield %48 : f16
        } -> tensor<?x?x?xf16>
        %36 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%35 : tensor<?x?x?xf16>) outs(%36 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %46 = arith.addf %in, %out : f16
          linalg.yield %46 : f16
        } -> tensor<?x?x1xf16>
        %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %32 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %46 = arith.subf %in, %in_6 : f16
          %47 = math.exp %46 : f16
          linalg.yield %47 : f16
        } -> tensor<?x?x1xf16>
        %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %38, %37 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%5 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
          %46 = arith.mulf %in, %in_6 : f16
          %47 = arith.addf %46, %in_7 : f16
          linalg.yield %47 : f16
        } -> tensor<?x?x1xf16>
        %subview_5 = memref.subview %arg1[%7, %23, 0] [%0, %2, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %40 = bufferization.to_tensor %subview_5 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        %41 = tensor.empty(%0, %2) : tensor<?x?x128xf16>
        %42 = loom.sync ins(%40 : tensor<?x?x128xf16>) outs(%41 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %43 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %44 = linalg.batch_matmul ins(%35, %42 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%43 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %arg9, %38 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%6 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
          %46 = arith.mulf %in_6, %in_7 : f16
          %47 = arith.addf %in, %46 : f16
          linalg.yield %47 : f16
        } -> tensor<?x?x128xf16>
        scf.yield %32, %39, %45 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
      }
      %17 = tensor.empty(%0, %1) : tensor<?x?x32xf16>
      %18 = loom.broadcast ins(%16#1 : tensor<?x?x1xf16>) outs(%17 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
      %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16#2, %18 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%6 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %23 = arith.divf %in, %in_4 : f16
        linalg.yield %23 : f16
      } -> tensor<?x?x128xf16>
      %subview_3 = memref.subview %arg3[%7, %8, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %20 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
      %21 = loom.sync ins(%19 : tensor<?x?x128xf16>) outs(%20 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %22 = bufferization.to_buffer %21 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      memref.copy %22, %subview_3 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
    }
    return
  }
}
