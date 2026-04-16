module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.442380e+00 : f16
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_h {upper_bound = 16 : index} : index
    %4 = loom.sym @tile_b {upper_bound = 2 : index} : index
    %5 = loom.sym @tile_c {upper_bound = 8 : index} : index
    %6 = arith.ceildivui %c16, %3 : index
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
      %subview = memref.subview %arg1[%12, %13, %14, %15] [1, 1, 1, %0] [1, 1, 1, 1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %16 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %17 = tensor.empty(%0) : tensor<?xf16>
      %18 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%16 : tensor<?xf16>) outs(%17 : tensor<?xf16>) {
      ^bb0(%in: f16, %out: f16):
        %36 = arith.mulf %in, %cst_1 : f16
        %37 = math.powf %cst, %36 : f16
        linalg.yield %37 : f16
      } -> tensor<?xf16>
      %19 = arith.muli %14, %c256 : index
      %20 = arith.divui %13, %c16 : index
      %subview_2 = memref.subview %arg4[%12, %19, %20, 0] [1, %0, 1, 16] [1, 1, 1, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %21 = bufferization.to_tensor %subview_2 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %22 = arith.muli %arg10, %1 : index
      %subview_3 = memref.subview %arg5[%12, %14, %13, %22, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %23 = bufferization.to_tensor %subview_3 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %24 = tensor.empty(%1) : tensor<16x?xf16>
      %transposed = linalg.transpose ins(%23 : tensor<?x16xf16>) outs(%24 : tensor<16x?xf16>) permutation = [1, 0] 
      %25 = linalg.fill ins(%cst_0 : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %26 = linalg.matmul ins(%21, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%26, %18 : tensor<?x?xf16>, tensor<?xf16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_7: f16, %out: f16):
        %36 = arith.mulf %in, %in_7 : f16
        linalg.yield %36 : f16
      } -> tensor<?x?xf16>
      %28 = arith.addi %arg9, %c1 : index
      %29 = arith.muli %28, %0 : index
      %30 = arith.ceildivui %29, %2 : index
      %31 = scf.for %arg13 = %c0 to %30 step %c1 iter_args(%arg14 = %27) -> (tensor<?x?xf16>) {
        %36 = arith.muli %arg13, %2 : index
        %subview_7 = memref.subview %arg0[%12, %14, %20, %15, %36] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %37 = bufferization.to_tensor %subview_7 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_8 = memref.subview %arg1[%12, %13, %14, %36] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %38 = bufferization.to_tensor %subview_8 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %39 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %subview_9 = memref.subview %arg2[%12, %13, %14, %36] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %40 = bufferization.to_tensor %subview_9 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%37, %16, %38, %40 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%39 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f16, %in_12: f16, %in_13: f16, %out: f16):
          %44 = arith.mulf %in_12, %cst_1 : f16
          %45 = arith.mulf %in_11, %cst_1 : f16
          %46 = arith.subf %45, %44 : f16
          %47 = math.powf %cst, %46 : f16
          %48 = arith.mulf %in, %47 : f16
          %49 = arith.mulf %48, %in_13 : f16
          linalg.yield %49 : f16
        } -> tensor<?x?xf16>
        %subview_10 = memref.subview %arg3[%12, %19, %13, %22] [1, %2, 1, %1] [1, 1, 1, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %42 = bufferization.to_tensor %subview_10 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
        %43 = linalg.matmul ins(%41, %42 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) -> tensor<?x?xf16>
        scf.yield %43 : tensor<?x?xf16>
      }
      %subview_4 = memref.subview %arg6[%13] [1] [1] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      %32 = bufferization.to_tensor %subview_4 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %subview_5 = memref.subview %arg3[%12, %19, %13, %22] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %33 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
      %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %33, %32 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
        %36 = arith.mulf %in_7, %in_8 : f16
        %37 = arith.addf %in, %36 : f16
        linalg.yield %37 : f16
      } -> tensor<?x?xf16>
      %subview_6 = memref.subview %arg7[%12, %19, %13, %22] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %35 = bufferization.to_buffer %34 : tensor<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      memref.copy %35, %subview_6 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    }
    return
  }
}
