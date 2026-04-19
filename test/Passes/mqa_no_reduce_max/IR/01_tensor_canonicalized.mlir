module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
    %cst = arith.constant 2.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %cst_2 = arith.constant 0xFC00 : f16
    %cst_3 = arith.constant 1.275630e-01 : f16
    %c8192 = arith.constant 8192 : index
    %c16 = arith.constant 16 : index
    %0 = loom.sym @tile_b {upper_bound = 16 : index} : index
    %1 = loom.sym @tile_s {upper_bound = 8192 : index} : index
    %2 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %3 = arith.ceildivui %c16, %0 : index
    %4 = arith.ceildivui %c8192, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0) : tensor<?x32xf16>
      %6 = tensor.empty(%0) : tensor<?x32x128xf16>
      %7 = arith.muli %arg4, %0 : index
      %subview = memref.subview %arg3[%7, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      %8 = bufferization.to_tensor %subview : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to tensor<?x32x128xf16>
      %9 = arith.muli %arg5, %1 : index
      %10 = arith.ceildivui %1, %2 : index
      %11 = linalg.fill ins(%cst_0 : f16) outs(%6 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %12 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %13 = linalg.fill ins(%cst_2 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %14:3 = scf.for %arg6 = %c0 to %10 step %c1 iter_args(%arg7 = %13, %arg8 = %12, %arg9 = %11) -> (tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>) {
        %18 = arith.muli %arg6, %2 : index
        %19 = arith.addi %9, %18 : index
        %subview_4 = memref.subview %arg0[%7, 0, %19] [%0, 128, %2] [1, 1, 1] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        %20 = bufferization.to_tensor %subview_4 : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to tensor<?x128x?xf16>
        %21 = tensor.empty(%0, %2) : tensor<?x32x?xf16>
        %22 = linalg.fill ins(%cst_0 : f16) outs(%21 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %23 = linalg.batch_matmul ins(%8, %20 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%22 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %24 = linalg.fill ins(%cst_2 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%23 : tensor<?x32x?xf16>) outs(%24 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %36 = arith.maximumf %in, %out : f16
          linalg.yield %36 : f16
        } -> tensor<?x32xf16>
        %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %25 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %36 = arith.mulf %in_6, %cst_3 : f16
          %37 = arith.cmpf ogt, %in, %36 : f16
          %38 = arith.select %37, %in, %36 : f16
          linalg.yield %38 : f16
        } -> tensor<?x32xf16>
        %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23, %26 : tensor<?x32x?xf16>, tensor<?x32xf16>) outs(%21 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %36 = arith.mulf %in, %cst_3 : f16
          %37 = arith.subf %36, %in_6 : f16
          %38 = math.powf %cst, %37 : f16
          linalg.yield %38 : f16
        } -> tensor<?x32x?xf16>
        %28 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%27 : tensor<?x32x?xf16>) outs(%28 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %36 = arith.addf %in, %out : f16
          linalg.yield %36 : f16
        } -> tensor<?x32xf16>
        %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %26 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_6: f16, %out: f16):
          %36 = arith.subf %in, %in_6 : f16
          %37 = math.powf %cst, %36 : f16
          linalg.yield %37 : f16
        } -> tensor<?x32xf16>
        %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %30, %29 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32xf16>) outs(%5 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
          %36 = arith.mulf %in, %in_6 : f16
          %37 = arith.addf %36, %in_7 : f16
          linalg.yield %37 : f16
        } -> tensor<?x32xf16>
        %subview_5 = memref.subview %arg1[%7, %19, 0] [%0, %2, 128] [1, 1, 1] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        %32 = bufferization.to_tensor %subview_5 : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        %33 = linalg.fill ins(%cst_0 : f16) outs(%6 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %34 = linalg.batch_matmul ins(%27, %32 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%33 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%34, %arg9, %30 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32xf16>) outs(%6 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
          %36 = arith.mulf %in_6, %in_7 : f16
          %37 = arith.addf %in, %36 : f16
          linalg.yield %37 : f16
        } -> tensor<?x32x128xf16>
        scf.yield %26, %31, %35 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>
      }
      %15 = tensor.empty(%4, %0) : tensor<?x?x32x128xf16>
      %16 = loom.gather ins(%14#2 : tensor<?x32x128xf16>) outs(%15 : tensor<?x?x32x128xf16>) across(%arg5 : index) -> tensor<?x?x32x128xf16>
      %17 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %17 {
        %18 = linalg.fill ins(%cst_0 : f16) outs(%6 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%16 : tensor<?x?x32x128xf16>) outs(%18 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %21 = arith.addf %in, %out : f16
          linalg.yield %21 : f16
        } -> tensor<?x32x128xf16>
        %subview_4 = memref.subview %arg2[%7, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %20 = bufferization.to_buffer %19 : tensor<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        memref.copy %20, %subview_4 : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      }
    }
    return
  }
}
