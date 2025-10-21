#map = affine_map<()[s0] -> (s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1) -> (d0, 0)>
module {
  func.func @flashattn_fwd(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index {tt.divisibility = 16 : i32}, %arg5: index {tt.divisibility = 16 : i32}, %arg6: index {tt.divisibility = 16 : i32}, %arg7: index {tt.divisibility = 16 : i32}, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index) {
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant 1.250000e-01 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %5 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %c64 = arith.constant 64 : index
    %6 = arith.muli %arg11, %c64 : index
    %7 = arith.muli %6, %arg4 : index
    %8 = affine.apply #map()[%7]
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%8], sizes: [64, 64], strides: [%arg4, 1] : memref<*xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>>
    %alloc = memref.alloc() : memref<64x64xf16>
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<64x64xf16>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16> to tensor<64x64xf16>
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c64_3 = arith.constant 64 : index
    %10:5 = scf.for %arg14 = %c0 to %c512 step %c64_3 iter_args(%arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %5, %arg18 = %1, %arg19 = %4) -> (i64, i64, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>) {
      %17 = arith.index_cast %arg16 : i64 to index
      %18 = affine.apply #map()[%17]
      %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%18], sizes: [64, 64], strides: [%arg5, 1] : memref<*xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>>
      %alloc_6 = memref.alloc() : memref<64x64xf16>
      memref.copy %reinterpret_cast_5, %alloc_6 : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<64x64xf16>
      %19 = bufferization.to_tensor %alloc_6 restrict writable : memref<64x64xf16> to tensor<64x64xf16>
      %20 = linalg.matmul ins(%9, %19 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %21 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%20, %3 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%20 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %47 = arith.mulf %in, %in_15 : f32
        linalg.yield %47 : f32
      } -> tensor<64x64xf32>
      %transposed = linalg.transpose ins(%21 : tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) permutation = [1, 0] 
      %reduced = linalg.reduce ins(%transposed : tensor<64x64xf32>) outs(%5 : tensor<64xf32>) dimensions = [0] 
        (%in: f32, %init: f32) {
          %47 = arith.maxnumf %in, %init : f32
          linalg.yield %47 : f32
        }
      %22 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg17, %reduced : tensor<64xf32>, tensor<64xf32>) outs(%arg17 : tensor<64xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %47 = arith.maxnumf %in, %in_15 : f32
        linalg.yield %47 : f32
      } -> tensor<64xf32>
      %expanded_7 = tensor.expand_shape %22 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %23 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_7 : tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<64x64xf32>
      %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%21, %23 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%21 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %47 = arith.subf %in, %in_15 : f32
        linalg.yield %47 : f32
      } -> tensor<64x64xf32>
      %25 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%24 : tensor<64x64xf32>) outs(%24 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %47 = math.exp %in : f32
        linalg.yield %47 : f32
      } -> tensor<64x64xf32>
      %transposed_8 = linalg.transpose ins(%25 : tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) permutation = [1, 0] 
      %26 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
      %reduced_9 = linalg.reduce ins(%transposed_8 : tensor<64x64xf32>) outs(%26 : tensor<64xf32>) dimensions = [0] 
        (%in: f32, %init: f32) {
          %47 = arith.addf %in, %init : f32
          linalg.yield %47 : f32
        }
      %27 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg17, %22 : tensor<64xf32>, tensor<64xf32>) outs(%arg17 : tensor<64xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %47 = arith.subf %in, %in_15 : f32
        linalg.yield %47 : f32
      } -> tensor<64xf32>
      %28 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%27 : tensor<64xf32>) outs(%27 : tensor<64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %47 = math.exp %in : f32
        linalg.yield %47 : f32
      } -> tensor<64xf32>
      %29 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg18, %28 : tensor<64xf32>, tensor<64xf32>) outs(%arg18 : tensor<64xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %47 = arith.mulf %in, %in_15 : f32
        linalg.yield %47 : f32
      } -> tensor<64xf32>
      %30 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%29, %reduced_9 : tensor<64xf32>, tensor<64xf32>) outs(%29 : tensor<64xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %47 = arith.addf %in, %in_15 : f32
        linalg.yield %47 : f32
      } -> tensor<64xf32>
      %31 = arith.index_cast %arg15 : i64 to index
      %32 = arith.muli %31, %arg6 : index
      %33 = affine.apply #map()[%32]
      %reinterpret_cast_10 = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [64, 64], strides: [%arg6, 1] : memref<*xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>>
      %alloc_11 = memref.alloc() : memref<64x64xf16>
      memref.copy %reinterpret_cast_10, %alloc_11 : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<64x64xf16>
      %34 = bufferization.to_tensor %alloc_11 restrict writable : memref<64x64xf16> to tensor<64x64xf16>
      %35 = tensor.empty() : tensor<64x64xf16>
      %36 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<64x64xf32>) outs(%35 : tensor<64x64xf16>) {
      ^bb0(%in: f32, %out: f16):
        %47 = arith.truncf %in : f32 to f16
        linalg.yield %47 : f16
      } -> tensor<64x64xf16>
      %expanded_12 = tensor.expand_shape %28 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
      %37 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded_12 : tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<64x64xf32>
      %38 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg19, %37 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg19 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %47 = arith.mulf %in, %in_15 : f32
        linalg.yield %47 : f32
      } -> tensor<64x64xf32>
      %39 = linalg.matmul ins(%36, %34 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32>
      %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%38, %39 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%38 : tensor<64x64xf32>) {
      ^bb0(%in: f32, %in_15: f32, %out: f32):
        %47 = arith.addf %in, %in_15 : f32
        linalg.yield %47 : f32
      } -> tensor<64x64xf32>
      %41 = arith.index_cast %arg15 : i64 to index
      %c64_13 = arith.constant 64 : index
      %42 = arith.addi %41, %c64_13 : index
      %43 = arith.index_cast %arg16 : i64 to index
      %c64_14 = arith.constant 64 : index
      %44 = arith.addi %43, %c64_14 : index
      %45 = arith.index_cast %42 : index to i64
      %46 = arith.index_cast %44 : index to i64
      scf.yield %45, %46, %22, %30, %40 : i64, i64, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>
    }
    %expanded = tensor.expand_shape %10#3 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<64x64xf32>
    %12 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%10#4, %11 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%10#4 : tensor<64x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %17 = arith.divf %in, %in_5 : f32
      linalg.yield %17 : f32
    } -> tensor<64x64xf32>
    %13 = tensor.empty() : tensor<64x64xf16>
    %14 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<64x64xf32>) outs(%13 : tensor<64x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %17 = arith.truncf %in : f32 to f16
      linalg.yield %17 : f16
    } -> tensor<64x64xf16>
    %15 = arith.muli %6, %arg7 : index
    %16 = affine.apply #map()[%15]
    %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%16], sizes: [64, 64], strides: [%arg7, 1] : memref<*xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>>
    bufferization.materialize_in_destination %14 in writable %reinterpret_cast_4 : (tensor<64x64xf16>, memref<64x64xf16, strided<[?, 1], offset: ?>>) -> ()
    return
  }
}

