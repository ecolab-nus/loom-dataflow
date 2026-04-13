module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
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
      %11 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %12 = arith.muli %arg11, %4 : index
      %13 = arith.muli %arg8, %3 : index
      %14 = arith.muli %arg12, %5 : index
      %15 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%12, %13, %14, %15] [1, 1, 1, %0] [1, 1, 1, 1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %16 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %17 = tensor.empty(%0) : tensor<?xf32>
      %18 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%16 : tensor<?xf16>) outs(%17 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %38 = arith.extf %in : f16 to f32
        linalg.yield %38 : f32
      } -> tensor<?xf32>
      %19 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%18 : tensor<?xf32>) outs(%17 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %38 = arith.truncf %cst_0 : f64 to f32
        %39 = arith.mulf %in, %38 : f32
        %40 = math.powf %cst, %39 : f32
        linalg.yield %40 : f32
      } -> tensor<?xf32>
      %20 = arith.muli %14, %c256 : index
      %21 = arith.divui %13, %c16 : index
      %subview_2 = memref.subview %arg4[%12, %20, %21, 0] [1, %0, 1, 16] [1, 1, 1, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %22 = bufferization.to_tensor %subview_2 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %23 = arith.muli %arg10, %1 : index
      %subview_3 = memref.subview %arg5[%12, %14, %13, %23, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %24 = bufferization.to_tensor %subview_3 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %25 = tensor.empty(%1) : tensor<16x?xf16>
      %transposed = linalg.transpose ins(%24 : tensor<?x16xf16>) outs(%25 : tensor<16x?xf16>) permutation = [1, 0] 
      %26 = linalg.fill ins(%cst_1 : f32) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %27 = linalg.matmul ins(%22, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%26 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%27, %19 : tensor<?x?xf32>, tensor<?xf32>) outs(%11 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %38 = arith.mulf %in, %in_7 : f32
        linalg.yield %38 : f32
      } -> tensor<?x?xf32>
      %29 = arith.addi %arg9, %c1 : index
      %30 = arith.muli %29, %0 : index
      %31 = arith.ceildivui %30, %2 : index
      %32 = scf.for %arg13 = %c0 to %31 step %c1 iter_args(%arg14 = %28) -> (tensor<?x?xf32>) {
        %38 = arith.muli %arg13, %2 : index
        %subview_7 = memref.subview %arg0[%12, %14, %21, %15, %38] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %39 = bufferization.to_tensor %subview_7 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_8 = memref.subview %arg1[%12, %13, %14, %38] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %40 = bufferization.to_tensor %subview_8 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %41 = tensor.empty(%2) : tensor<?xf32>
        %42 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%40 : tensor<?xf16>) outs(%41 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %49 = arith.extf %in : f16 to f32
          linalg.yield %49 : f32
        } -> tensor<?xf32>
        %subview_9 = memref.subview %arg2[%12, %13, %14, %38] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %43 = bufferization.to_tensor %subview_9 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %44 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%43 : tensor<?xf16>) outs(%41 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %49 = arith.extf %in : f16 to f32
          linalg.yield %49 : f32
        } -> tensor<?xf32>
        %45 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%39, %18, %42, %44 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%45 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f32, %in_12: f32, %in_13: f32, %out: f16):
          %49 = arith.truncf %cst_0 : f64 to f32
          %50 = arith.mulf %in_12, %49 : f32
          %51 = arith.truncf %cst_0 : f64 to f32
          %52 = arith.mulf %in_11, %51 : f32
          %53 = arith.subf %52, %50 : f32
          %54 = math.powf %cst, %53 : f32
          %55 = arith.extf %in : f16 to f32
          %56 = arith.mulf %55, %54 : f32
          %57 = arith.mulf %56, %in_13 : f32
          %58 = arith.truncf %57 : f32 to f16
          linalg.yield %58 : f16
        } -> tensor<?x?xf16>
        %subview_10 = memref.subview %arg3[%12, %20, %13, %23] [1, %2, 1, %1] [1, 1, 1, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %47 = bufferization.to_tensor %subview_10 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
        %48 = linalg.matmul ins(%46, %47 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        scf.yield %48 : tensor<?x?xf32>
      }
      %subview_4 = memref.subview %arg6[%13] [1] [1] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      %33 = bufferization.to_tensor %subview_4 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %subview_5 = memref.subview %arg3[%12, %20, %13, %23] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %34 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
      %35 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32, %34, %33 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%35 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %in_7: f16, %in_8: f16, %out: f16):
        %38 = arith.extf %in_8 : f16 to f32
        %39 = arith.extf %in_7 : f16 to f32
        %40 = arith.mulf %39, %38 : f32
        %41 = arith.addf %in, %40 : f32
        %42 = arith.truncf %41 : f32 to f16
        linalg.yield %42 : f16
      } -> tensor<?x?xf16>
      %subview_6 = memref.subview %arg7[%12, %20, %13, %23] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %37 = bufferization.to_buffer %36 : tensor<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      memref.copy %37, %subview_6 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    }
    return
  }
}
