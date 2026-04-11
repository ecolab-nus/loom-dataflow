module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_h {upper_bound = 16 : index} : index
    %4 = loom.sym @tile_c {upper_bound = 8 : index} : index
    %5 = arith.ceildivui %c16, %3 : index
    %6 = arith.ceildivui %c256, %0 : index
    %7 = arith.ceildivui %c64, %1 : index
    %8 = arith.ceildivui %c8, %4 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%5), symbol(%6), symbol(%7), 1, symbol(%8)) {
      %9 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %10 = arith.muli %arg8, %3 : index
      %11 = arith.muli %arg12, %4 : index
      %12 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%arg11, %10, %11, %12] [1, 1, 1, %0] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %13 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %14 = tensor.empty(%0) : tensor<?xf32>
      %15 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%13 : tensor<?xf16>) outs(%14 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %35 = arith.extf %in : f16 to f32
        linalg.yield %35 : f32
      } -> tensor<?xf32>
      %16 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%15 : tensor<?xf32>) outs(%14 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %35 = arith.truncf %cst_0 : f64 to f32
        %36 = arith.mulf %in, %35 : f32
        %37 = math.powf %cst, %36 : f32
        linalg.yield %37 : f32
      } -> tensor<?xf32>
      %17 = arith.muli %11, %c256 : index
      %18 = arith.divui %10, %c16 : index
      %subview_2 = memref.subview %arg4[%arg11, %17, %18, 0] [1, %0, 1, 16] [1, 1, 1, 1] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %19 = bufferization.to_tensor %subview_2 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %20 = arith.muli %arg10, %1 : index
      %subview_3 = memref.subview %arg5[%arg11, %11, %10, %20, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      %21 = bufferization.to_tensor %subview_3 : memref<?x16xf16, strided<[16, 1], offset: ?>> to tensor<?x16xf16>
      %22 = tensor.empty(%1) : tensor<16x?xf16>
      %transposed = linalg.transpose ins(%21 : tensor<?x16xf16>) outs(%22 : tensor<16x?xf16>) permutation = [1, 0] 
      %23 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %24 = linalg.matmul ins(%19, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%24, %16 : tensor<?x?xf32>, tensor<?xf32>) outs(%9 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %35 = arith.mulf %in, %in_7 : f32
        linalg.yield %35 : f32
      } -> tensor<?x?xf32>
      %26 = arith.addi %arg9, %c1 : index
      %27 = arith.muli %26, %0 : index
      %28 = arith.ceildivui %27, %2 : index
      %29 = scf.for %arg13 = %c0 to %28 step %c1 iter_args(%arg14 = %25) -> (tensor<?x?xf32>) {
        %35 = arith.muli %arg13, %2 : index
        %subview_7 = memref.subview %arg0[%arg11, %11, %18, %12, %35] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %36 = bufferization.to_tensor %subview_7 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_8 = memref.subview %arg1[%arg11, %10, %11, %35] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %37 = bufferization.to_tensor %subview_8 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %38 = tensor.empty(%2) : tensor<?xf32>
        %39 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%37 : tensor<?xf16>) outs(%38 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %46 = arith.extf %in : f16 to f32
          linalg.yield %46 : f32
        } -> tensor<?xf32>
        %subview_9 = memref.subview %arg2[%arg11, %10, %11, %35] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %40 = bufferization.to_tensor %subview_9 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %41 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%40 : tensor<?xf16>) outs(%38 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %46 = arith.extf %in : f16 to f32
          linalg.yield %46 : f32
        } -> tensor<?xf32>
        %42 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%36, %15, %39, %41 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%42 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_11: f32, %in_12: f32, %in_13: f32, %out: f16):
          %46 = arith.truncf %cst_0 : f64 to f32
          %47 = arith.mulf %in_12, %46 : f32
          %48 = arith.truncf %cst_0 : f64 to f32
          %49 = arith.mulf %in_11, %48 : f32
          %50 = arith.subf %49, %47 : f32
          %51 = math.powf %cst, %50 : f32
          %52 = arith.extf %in : f16 to f32
          %53 = arith.mulf %52, %51 : f32
          %54 = arith.mulf %53, %in_13 : f32
          %55 = arith.truncf %54 : f32 to f16
          linalg.yield %55 : f16
        } -> tensor<?x?xf16>
        %subview_10 = memref.subview %arg3[%arg11, %17, %10, %20] [1, %2, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %44 = bufferization.to_tensor %subview_10 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
        %45 = linalg.matmul ins(%43, %44 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        scf.yield %45 : tensor<?x?xf32>
      }
      %subview_4 = memref.subview %arg6[%10] [1] [1] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      %30 = bufferization.to_tensor %subview_4 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %subview_5 = memref.subview %arg3[%arg11, %17, %10, %20] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %31 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
      %32 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%29, %31, %30 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%32 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %in_7: f16, %in_8: f16, %out: f16):
        %35 = arith.extf %in_8 : f16 to f32
        %36 = arith.extf %in_7 : f16 to f32
        %37 = arith.mulf %36, %35 : f32
        %38 = arith.addf %in, %37 : f32
        %39 = arith.truncf %38 : f32 to f16
        linalg.yield %39 : f16
      } -> tensor<?x?xf16>
      %subview_6 = memref.subview %arg7[%arg11, %17, %10, %20] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %34 = bufferization.to_buffer %33 : tensor<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      memref.copy %34, %subview_6 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    }
    return
  }
}
