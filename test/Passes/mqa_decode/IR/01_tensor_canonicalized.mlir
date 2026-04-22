module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c8192 = arith.constant 8192 : index
    %c16 = arith.constant 16 : index
    %0 = loom.sym @tile_b {upper_bound = 16 : index} : index
    %1 = loom.sym @tile_s {upper_bound = 8192 : index} : index
    %2 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %3 = arith.ceildivui %c16, %0 : index
    %4 = arith.ceildivui %c8192, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0) : tensor<?x32x1xf16>
      %6 = tensor.empty(%0) : tensor<?x32x128xf16>
      %7 = arith.muli %arg4, %0 : index
      %subview = memref.subview %arg3[%7, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      %8 = bufferization.to_tensor %subview : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to tensor<?x32x128xf16>
      %9 = arith.muli %arg5, %1 : index
      %10 = arith.ceildivui %1, %2 : index
      %11 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %12 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %13 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %14:3 = scf.for %arg6 = %c0 to %10 step %c1 iter_args(%arg7 = %13, %arg8 = %12, %arg9 = %11) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %28 = arith.muli %arg6, %2 : index
        %29 = arith.addi %9, %28 : index
        %subview_3 = memref.subview %arg0[%7, 0, %29] [%0, 128, %2] [1, 1, 1] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        %30 = bufferization.to_tensor %subview_3 : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to tensor<?x128x?xf16>
        %31 = tensor.empty(%0, %2) : tensor<?x32x?xf16>
        %32 = linalg.fill ins(%cst : f16) outs(%31 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %33 = linalg.batch_matmul ins(%8, %30 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%32 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %34 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%33 : tensor<?x32x?xf16>) outs(%34 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %49 = arith.maximumf %in, %out : f16
          linalg.yield %49 : f16
        } -> tensor<?x32x1xf16>
        %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %35 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %49 = arith.mulf %in_5, %cst_2 : f16
          %50 = arith.cmpf ogt, %in, %49 : f16
          %51 = arith.select %50, %in, %49 : f16
          linalg.yield %51 : f16
        } -> tensor<?x32x1xf16>
        %37 = tensor.empty(%0) : tensor<?x32x32xf16>
        %38 = loom.broadcast ins(%36 : tensor<?x32x1xf16>) outs(%37 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
        %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %38 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%31 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %49 = arith.mulf %in, %cst_2 : f16
          %50 = arith.subf %49, %in_5 : f16
          %51 = math.exp %50 : f16
          linalg.yield %51 : f16
        } -> tensor<?x32x?xf16>
        %40 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%39 : tensor<?x32x?xf16>) outs(%40 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %49 = arith.addf %in, %out : f16
          linalg.yield %49 : f16
        } -> tensor<?x32x1xf16>
        %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %36 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %49 = arith.subf %in, %in_5 : f16
          %50 = math.exp %49 : f16
          linalg.yield %50 : f16
        } -> tensor<?x32x1xf16>
        %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %42, %41 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
          %49 = arith.mulf %in, %in_5 : f16
          %50 = arith.addf %49, %in_6 : f16
          linalg.yield %50 : f16
        } -> tensor<?x32x1xf16>
        %44 = loom.broadcast ins(%42 : tensor<?x32x1xf16>) outs(%37 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
        %subview_4 = memref.subview %arg1[%7, %29, 0] [%0, %2, 128] [1, 1, 1] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        %45 = bufferization.to_tensor %subview_4 : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        %46 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %47 = linalg.batch_matmul ins(%39, %45 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%46 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %arg9, %44 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%6 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
          %49 = arith.mulf %in_5, %in_6 : f16
          %50 = arith.addf %in, %49 : f16
          linalg.yield %50 : f16
        } -> tensor<?x32x128xf16>
        scf.yield %36, %43, %48 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#1, %14#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %28 = math.log %in : f16
        %29 = arith.addf %28, %in_3 : f16
        linalg.yield %29 : f16
      } -> tensor<?x32x1xf16>
      %16 = tensor.empty(%0) : tensor<?x32x32xf16>
      %17 = loom.broadcast ins(%14#1 : tensor<?x32x1xf16>) outs(%16 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
      %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#2, %17 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%6 : tensor<?x32x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %28 = arith.divf %in, %in_3 : f16
        linalg.yield %28 : f16
      } -> tensor<?x32x128xf16>
      %19 = tensor.empty(%4, %0) : tensor<?x?x32x1xf16>
      %20 = tensor.empty(%0) : tensor<?x32x1xf16>
      %21 = loom.sync ins(%15 : tensor<?x32x1xf16>) outs(%20 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %22 = loom.gather ins(%21 : tensor<?x32x1xf16>) outs(%19 : tensor<?x?x32x1xf16>) across(%arg5 : index) -> tensor<?x?x32x1xf16>
      %23 = tensor.empty(%4, %0) : tensor<?x?x32x128xf16>
      %24 = tensor.empty(%0) : tensor<?x32x128xf16>
      %25 = loom.sync ins(%18 : tensor<?x32x128xf16>) outs(%24 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %26 = loom.gather ins(%25 : tensor<?x32x128xf16>) outs(%23 : tensor<?x?x32x128xf16>) across(%arg5 : index) -> tensor<?x?x32x128xf16>
      %27 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %27 {
        %28 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %29 = tensor.empty(%4, %0) : tensor<?x?x32x1xf16>
        %30 = loom.sync ins(%22 : tensor<?x?x32x1xf16>) outs(%29 : tensor<?x?x32x1xf16>) -> tensor<?x?x32x1xf16>
        %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%30 : tensor<?x?x32x1xf16>) outs(%28 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %47 = arith.maximumf %in, %out : f16
          linalg.yield %47 : f16
        } -> tensor<?x32x1xf16>
        %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%30, %31 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%19 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %47 = arith.subf %in, %in_4 : f16
          %48 = math.exp %47 : f16
          linalg.yield %48 : f16
        } -> tensor<?x?x32x1xf16>
        %33 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%32 : tensor<?x?x32x1xf16>) outs(%33 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %47 = arith.addf %in, %out : f16
          linalg.yield %47 : f16
        } -> tensor<?x32x1xf16>
        %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32, %34 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%19 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %47 = arith.divf %in, %in_4 : f16
          linalg.yield %47 : f16
        } -> tensor<?x?x32x1xf16>
        %36 = tensor.empty(%4, %0) : tensor<?x?x32x32xf16>
        %37 = loom.broadcast ins(%35 : tensor<?x?x32x1xf16>) outs(%36 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
        %38 = arith.cmpi eq, %4, %1 : index
        %39 = tensor.empty(%4, %0) : tensor<?x?x32x128xf16>
        %40 = loom.sync ins(%26 : tensor<?x?x32x128xf16>) outs(%39 : tensor<?x?x32x128xf16>) -> tensor<?x?x32x128xf16>
        %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%40, %37 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%23 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %47 = arith.mulf %in, %in_4 : f16
          linalg.yield %47 : f16
        } -> tensor<?x?x32x128xf16>
        %42 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%41 : tensor<?x?x32x128xf16>) outs(%42 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %47 = arith.addf %in, %out : f16
          linalg.yield %47 : f16
        } -> tensor<?x32x128xf16>
        %subview_3 = memref.subview %arg2[%7, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %44 = tensor.empty(%0) : tensor<?x32x128xf16>
        %45 = loom.sync ins(%43 : tensor<?x32x128xf16>) outs(%44 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %46 = bufferization.to_buffer %45 : tensor<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        memref.copy %46, %subview_3 : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      }
    }
    return
  }
}
