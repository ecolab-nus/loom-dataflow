module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x?xf16>, %arg5: memref<1x8x16x64x?xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_h {upper_bound = 16 : index} : index
    %4 = loom.sym @tile_c {upper_bound = 8 : index} : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (16 ceildiv symbol(%3), 256 ceildiv symbol(%0), 64 ceildiv symbol(%1), 1, 8 ceildiv symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %6 = arith.muli %arg8, %3 : index
      %7 = arith.muli %arg12, %4 : index
      %8 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%arg11, %6, %7, %8] [1, 1, 1, %0] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %9 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %10 = tensor.empty(%0) : tensor<?xf32>
      %11 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%9 : tensor<?xf16>) outs(%10 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %31 = arith.extf %in : f16 to f32
        linalg.yield %31 : f32
      } -> tensor<?xf32>
      %12 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%11 : tensor<?xf32>) outs(%10 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %31 = arith.truncf %cst_0 : f64 to f32
        %32 = arith.mulf %in, %31 : f32
        %33 = math.powf %cst, %32 : f32
        linalg.yield %33 : f32
      } -> tensor<?xf32>
      %13 = arith.muli %7, %c256 : index
      %14 = affine.apply affine_map<(d0) -> (d0 floordiv 16)>(%6)
      %dim = memref.dim %arg4, %c3 : memref<1x2048x1x?xf16>
      %subview_2 = memref.subview %arg4[%arg11, %13, %14, 0] [1, %0, 1, %dim] [1, 1, 1, 1] : memref<1x2048x1x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %15 = bufferization.to_tensor %subview_2 : memref<?x?xf16, strided<[?, 1], offset: ?>> to tensor<?x?xf16>
      %16 = arith.muli %arg10, %1 : index
      %dim_3 = memref.dim %arg5, %c4 : memref<1x8x16x64x?xf16>
      %subview_4 = memref.subview %arg5[%arg11, %7, %6, %16, 0] [1, 1, 1, %1, %dim_3] [1, 1, 1, 1, 1] : memref<1x8x16x64x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
      %17 = bufferization.to_tensor %subview_4 : memref<?x?xf16, strided<[?, 1], offset: ?>> to tensor<?x?xf16>
      %18 = tensor.empty(%dim_3, %1) : tensor<?x?xf16>
      %transposed = linalg.transpose ins(%17 : tensor<?x?xf16>) outs(%18 : tensor<?x?xf16>) permutation = [1, 0] 
      %19 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %20 = linalg.matmul ins(%15, %transposed : tensor<?x?xf16>, tensor<?x?xf16>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20, %12 : tensor<?x?xf32>, tensor<?xf32>) outs(%5 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_8: f32, %out: f32):
        %31 = arith.mulf %in, %in_8 : f32
        linalg.yield %31 : f32
      } -> tensor<?x?xf32>
      %22 = arith.addi %arg9, %c1 : index
      %23 = arith.muli %22, %0 : index
      %24 = affine.apply affine_map<(d0)[s0] -> (d0 ceildiv s0)>(%23)[%2]
      %25 = scf.for %arg13 = %c0 to %24 step %c1 iter_args(%arg14 = %21) -> (tensor<?x?xf32>) {
        %31 = arith.muli %arg13, %2 : index
        %subview_8 = memref.subview %arg0[%arg11, %7, %14, %8, %31] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %32 = bufferization.to_tensor %subview_8 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %subview_9 = memref.subview %arg1[%arg11, %6, %7, %31] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %33 = bufferization.to_tensor %subview_9 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %34 = tensor.empty(%2) : tensor<?xf32>
        %35 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%33 : tensor<?xf16>) outs(%34 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %42 = arith.extf %in : f16 to f32
          linalg.yield %42 : f32
        } -> tensor<?xf32>
        %subview_10 = memref.subview %arg2[%arg11, %6, %7, %31] [1, 1, 1, %2] [1, 1, 1, 1] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %36 = bufferization.to_tensor %subview_10 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%36 : tensor<?xf16>) outs(%34 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %42 = arith.extf %in : f16 to f32
          linalg.yield %42 : f32
        } -> tensor<?xf32>
        %38 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32, %11, %35, %37 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%38 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_12: f32, %in_13: f32, %in_14: f32, %out: f16):
          %42 = arith.truncf %cst_0 : f64 to f32
          %43 = arith.mulf %in_13, %42 : f32
          %44 = arith.truncf %cst_0 : f64 to f32
          %45 = arith.mulf %in_12, %44 : f32
          %46 = arith.subf %45, %43 : f32
          %47 = math.powf %cst, %46 : f32
          %48 = arith.extf %in : f16 to f32
          %49 = arith.mulf %48, %47 : f32
          %50 = arith.mulf %49, %in_14 : f32
          %51 = arith.truncf %50 : f32 to f16
          linalg.yield %51 : f16
        } -> tensor<?x?xf16>
        %subview_11 = memref.subview %arg3[%arg11, %13, %6, %16] [1, %2, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        %40 = bufferization.to_tensor %subview_11 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
        %41 = linalg.matmul ins(%39, %40 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        scf.yield %41 : tensor<?x?xf32>
      }
      %subview_5 = memref.subview %arg6[%6] [1] [1] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      %26 = bufferization.to_tensor %subview_5 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %subview_6 = memref.subview %arg3[%arg11, %13, %6, %16] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %27 = bufferization.to_tensor %subview_6 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to tensor<?x?xf16>
      %28 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%25, %27, %26 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%28 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %in_8: f16, %in_9: f16, %out: f16):
        %31 = arith.extf %in_9 : f16 to f32
        %32 = arith.extf %in_8 : f16 to f32
        %33 = arith.mulf %32, %31 : f32
        %34 = arith.addf %in, %33 : f32
        %35 = arith.truncf %34 : f32 to f16
        linalg.yield %35 : f16
      } -> tensor<?x?xf16>
      %subview_7 = memref.subview %arg7[%arg11, %13, %6, %16] [1, %0, 1, %1] [1, 1, 1, 1] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %30 = bufferization.to_buffer %29 : tensor<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      memref.copy %30, %subview_7 : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
    }
    return
  }
}
