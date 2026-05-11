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
      %subview = memref.subview %arg2[%7, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      %8 = loom.bufferize_to_tensor %subview[%0, 32, 128] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> -> tensor<?x32x128xf16>
      %9 = arith.muli %arg5, %1 : index
      %10 = arith.ceildivui %1, %2 : index
      %11 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %12 = linalg.fill ins(%cst_0 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %13 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %14:3 = scf.for %arg6 = %c0 to %10 step %c1 iter_args(%arg7 = %13, %arg8 = %12, %arg9 = %11) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %25 = arith.muli %arg6, %2 : index
        %26 = arith.addi %9, %25 : index
        %subview_3 = memref.subview %arg0[%7, 0, %26] [%0, 128, %2] [1, 1, 1] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        %27 = loom.bufferize_to_tensor %subview_3[%0, 128, %2] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> -> tensor<?x128x?xf16>
        %28 = tensor.empty(%0, %2) : tensor<?x32x?xf16>
        %29 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %30 = linalg.batch_matmul ins(%8, %27 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%29 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %31 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%30 : tensor<?x32x?xf16>) outs(%31 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %45 = arith.maximumf %in, %out : f16
          linalg.yield %45 : f16
        } -> tensor<?x32x1xf16>
        %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %32 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %45 = arith.mulf %in_5, %cst_2 : f16
          %46 = arith.cmpf ogt, %in, %45 : f16
          %47 = arith.select %46, %in, %45 : f16
          linalg.yield %47 : f16
        } -> tensor<?x32x1xf16>
        %34 = loom.broadcast ins(%33 : tensor<?x32x1xf16>) outs(%28 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
        %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30, %34 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%28 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %45 = arith.mulf %in, %cst_2 : f16
          %46 = arith.subf %45, %in_5 : f16
          %47 = math.exp %46 : f16
          linalg.yield %47 : f16
        } -> tensor<?x32x?xf16>
        %36 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%35 : tensor<?x32x?xf16>) outs(%36 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %45 = arith.addf %in, %out : f16
          linalg.yield %45 : f16
        } -> tensor<?x32x1xf16>
        %38 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %33 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %45 = arith.subf %in, %in_5 : f16
          %46 = math.exp %45 : f16
          linalg.yield %46 : f16
        } -> tensor<?x32x1xf16>
        %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %38, %37 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
          %45 = arith.mulf %in, %in_5 : f16
          %46 = arith.addf %45, %in_6 : f16
          linalg.yield %46 : f16
        } -> tensor<?x32x1xf16>
        %40 = loom.broadcast ins(%38 : tensor<?x32x1xf16>) outs(%6 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
        %subview_4 = memref.subview %arg1[%7, %26, 0] [%0, %2, 128] [1, 1, 1] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        %41 = loom.bufferize_to_tensor %subview_4[%0, %2, 128] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> -> tensor<?x?x128xf16>
        %42 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %43 = linalg.batch_matmul ins(%35, %41 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%42 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%43, %arg9, %40 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%6 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
          %45 = arith.mulf %in_5, %in_6 : f16
          %46 = arith.addf %in, %45 : f16
          linalg.yield %46 : f16
        } -> tensor<?x32x128xf16>
        scf.yield %33, %39, %44 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#1, %14#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%5 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %25 = math.log %in : f16
        %26 = arith.addf %25, %in_3 : f16
        linalg.yield %26 : f16
      } -> tensor<?x32x1xf16>
      %16 = loom.broadcast ins(%14#1 : tensor<?x32x1xf16>) outs(%6 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
      %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14#2, %16 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%6 : tensor<?x32x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %25 = arith.divf %in, %in_3 : f16
        linalg.yield %25 : f16
      } -> tensor<?x32x128xf16>
      %18 = loom.bufferize_to_memref %15 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
      %19 = loom.placeholder [%4, %0, 32, 1] : memref<?x?x32x1xf16>
      loom.gather %18, %19 across(%arg5 : index), area : [1, 1] : memref<?x32x1xf16> to memref<?x?x32x1xf16>
      %20 = loom.bufferize_to_tensor %19[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %21 = loom.bufferize_to_memref %17 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
      %22 = loom.placeholder [%4, %0, 32, 128] : memref<?x?x32x128xf16>
      loom.gather %21, %22 across(%arg5 : index), area : [1, 1] : memref<?x32x128xf16> to memref<?x?x32x128xf16>
      %23 = loom.bufferize_to_tensor %22[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %24 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %24 {
        %25 = linalg.fill ins(%cst_1 : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%20 : tensor<?x?x32x1xf16>) outs(%25 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %39 = arith.maximumf %in, %out : f16
          linalg.yield %39 : f16
        } -> tensor<?x32x1xf16>
        %27 = tensor.empty(%4, %0) : tensor<?x?x32x1xf16>
        %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%20, %26 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%27 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %39 = arith.subf %in, %in_4 : f16
          %40 = math.exp %39 : f16
          linalg.yield %40 : f16
        } -> tensor<?x?x32x1xf16>
        %29 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%28 : tensor<?x?x32x1xf16>) outs(%29 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %39 = arith.addf %in, %out : f16
          linalg.yield %39 : f16
        } -> tensor<?x32x1xf16>
        %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28, %30 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%27 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %39 = arith.divf %in, %in_4 : f16
          linalg.yield %39 : f16
        } -> tensor<?x?x32x1xf16>
        %32 = tensor.empty(%4, %0) : tensor<?x?x32x128xf16>
        %33 = loom.broadcast ins(%31 : tensor<?x?x32x1xf16>) outs(%32 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
        %34 = arith.cmpi eq, %4, %1 : index
        %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%23, %33 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%32 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %39 = arith.mulf %in, %in_4 : f16
          linalg.yield %39 : f16
        } -> tensor<?x?x32x128xf16>
        %36 = linalg.fill ins(%cst : f16) outs(%6 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%35 : tensor<?x?x32x128xf16>) outs(%36 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %39 = arith.addf %in, %out : f16
          linalg.yield %39 : f16
        } -> tensor<?x32x128xf16>
        %subview_3 = memref.subview %arg3[%7, 0, 0] [%0, 32, 128] [1, 1, 1] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %38 = loom.bufferize_to_memref %37 : tensor<?x32x128xf16> -> memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        memref.copy %38, %subview_3 : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      }
    }
    return
  }
}
