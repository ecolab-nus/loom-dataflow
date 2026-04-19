module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.442380e+00 : f16
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_h {upper_bound = 64 : index} : index
    %4 = loom.sym @tile_b {upper_bound = 2 : index} : index
    %5 = loom.sym @tile_c {upper_bound = 8 : index} : index
    %6 = arith.ceildivui %c64, %3 : index
    %7 = arith.ceildivui %c256, %0 : index
    %8 = arith.ceildivui %c64, %1 : index
    %9 = arith.ceildivui %c2, %4 : index
    %10 = arith.ceildivui %c8, %5 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%6), symbol(%7), symbol(%8), symbol(%9), symbol(%10)) {
      %11 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %12 = arith.muli %arg11, %4 : index
      %13 = arith.muli %arg8, %3 : index
      %14 = arith.muli %arg12, %5 : index
      %15 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%12, %13, %14, %15] [1, 1, 1, %0] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %16 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %17 = arith.muli %14, %c256 : index
      %18 = arith.divui %13, %c64 : index
      %subview_2 = memref.subview %arg4[%12, %17, %18, 0] [1, %0, 1, 64] [1, 1, 1, 1] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %19 = bufferization.to_tensor %subview_2 : memref<?x64xf16, strided<[64, 1], offset: ?>> to tensor<?x64xf16>
      %20 = arith.muli %arg10, %1 : index
      %subview_3 = memref.subview %arg5[%12, %14, %13, %20, 0] [1, 1, 1, %1, 64] [1, 1, 1, 1, 1] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %21 = bufferization.to_tensor %subview_3 : memref<?x64xf16, strided<[64, 1], offset: ?>> to tensor<?x64xf16>
      %22 = tensor.empty(%1) : tensor<64x?xf16>
      %transposed = linalg.transpose ins(%21 : tensor<?x64xf16>) outs(%22 : tensor<64x?xf16>) permutation = [1, 0] 
      %23 = linalg.fill ins(%cst_0 : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %24 = linalg.matmul ins(%19, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%24, %16 : tensor<?x?xf16>, tensor<?xf16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_7: f16, %out: f16):
        %34 = arith.mulf %in_7, %cst_1 : f16
        %35 = math.powf %cst, %34 : f16
        %36 = arith.mulf %in, %35 : f16
        linalg.yield %36 : f16
      } -> tensor<?x?xf16>
      %26 = arith.addi %arg9, %c1 : index
      %27 = arith.muli %26, %0 : index
      %28 = arith.ceildivui %27, %2 : index
      %29 = scf.for %arg13 = %c0 to %28 step %c1 iter_args(%arg14 = %25) -> (tensor<?x?xf16>) {
        %34 = arith.muli %arg13, %2 : index
        %subview_7 = memref.subview %arg0[%12, %14, %18, %15, %34] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %35 = bufferization.to_tensor %subview_7 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_8 = memref.subview %arg1[%12, %13, %14, %34] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %36 = bufferization.to_tensor %subview_8 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %37 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %subview_9 = memref.subview %arg2[%12, %13, %14, %34] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %38 = bufferization.to_tensor %subview_9 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%35, %16, %36, %38 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%37 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
          %44 = arith.mulf %in_12, %cst_1 : f16
          %45 = arith.mulf %in_11, %cst_1 : f16
          %46 = arith.subf %45, %44 : f16
          %47 = math.powf %cst, %46 : f16
          %48 = arith.mulf %in, %47 : f16
          %49 = arith.mulf %48, %in_13 : f16
          linalg.yield %49 : f16
        } -> tensor<?x?xf16>
        %subview_10 = memref.subview %arg3[%12, %17, %13, %20] [1, %2, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        %40 = bufferization.to_tensor %subview_10 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
        %41 = linalg.fill ins(%cst_0 : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %42 = linalg.matmul ins(%39, %40 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%41 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %42 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %out: f16):
          %44 = arith.addf %in, %in_11 : f16
          linalg.yield %44 : f16
        } -> tensor<?x?xf16>
        scf.yield %43 : tensor<?x?xf16>
      }
      %subview_4 = memref.subview %arg6[%13] [1] [1] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      %30 = bufferization.to_tensor %subview_4 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %subview_5 = memref.subview %arg3[%12, %17, %13, %20] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %31 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
      %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%29, %31, %30 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
        %34 = arith.mulf %in_7, %in_8 : f16
        %35 = arith.addf %in, %34 : f16
        linalg.yield %35 : f16
      } -> tensor<?x?xf16>
      %subview_6 = memref.subview %arg7[%12, %17, %13, %20] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %33 = bufferization.to_buffer %32 : tensor<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      memref.copy %33, %subview_6 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
    }
    return
  }
}
