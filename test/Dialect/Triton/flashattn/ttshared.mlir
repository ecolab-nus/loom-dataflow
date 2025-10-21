#loc = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0)
#loc1 = loc(unknown)
#loc2 = loc("/home/zhenyu/triton/python/triton/language/standard.py":188:40)
#loc3 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":80:38)
#loc4 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":77:29)
#loc14 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":72:35)
#loc15 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":77:40)
#loc16 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":80:31)
#loc17 = loc("/tmp/tmp_88odyst/tt.mlir":46:109)
#loc18 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":81:23)
#loc19 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":81:18)
#loc20 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":82:19)
#loc21 = loc("/home/zhenyu/triton/python/triton/language/standard.py":290:36)
#loc22 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":83:22)
#loc23 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":84:29)
#loc24 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":84:23)
#loc25 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":85:20)
#loc26 = loc("/tmp/tmp_88odyst/tt.mlir":46:126)
#loc28 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":88:17)
#loc29 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":89:30)
#loc30 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":89:24)
#loc31 = loc("/tmp/tmp_88odyst/tt.mlir":46:141)
#loc32 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":90:35)
#loc36 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":99:24)
#loc37 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":99:20)
#loc38 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":100:29)
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#loc39 = loc(callsite(#loc2 at #loc3))
#loc45 = loc(callsite(#loc21 at #loc22))
module {
  func.func @flashattn_fwd(%arg0: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg1: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg2: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg3: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg4: i32 {tt.divisibility = 16 : i32} loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg5: i32 {tt.divisibility = 16 : i32} loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg6: i32 {tt.divisibility = 16 : i32} loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg7: i32 {tt.divisibility = 16 : i32} loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg8: i32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg9: i32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg10: i32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg11: i32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg12: i32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg13: i32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":10:0)) {
    %cst = arith.constant 1.000000e+00 : f32 loc(#loc1)
    %c0_i64 = arith.constant 0 : i64 loc(#loc1)
    %c64_i64 = arith.constant 64 : i64 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c512_i32 = arith.constant 512 : i32 loc(#loc1)
    %cst_0 = arith.constant 1.250000e-01 : f32 loc(#loc1)
    %cst_1 = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %cst_2 = arith.constant 0xFF800000 : f32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %0 = tensor.empty() : tensor<64xf32> loc(#loc39)
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32> loc(#loc1)
    %2 = tensor.empty() : tensor<64x64xf32> loc(#loc4)
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32> loc(#loc1)
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32> loc(#loc4)
    %5 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32> loc(#loc39)
    %6 = arith.muli %arg11, %c64_i32 : i32 loc(#loc5)
    %7 = arith.index_cast %arg4 : i32 to index loc(#loc40)
    %8 = arith.index_cast %6 : i32 to index loc(#loc41)
    %9 = arith.index_cast %arg6 : i32 to index loc(#loc42)
    %10 = arith.index_cast %arg5 : i32 to index loc(#loc43)
    %11 = arith.index_cast %arg7 : i32 to index loc(#loc44)
    %12 = arith.muli %8, %7 : index loc(#loc6)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%12], sizes: [64, 64], strides: [%7, 1] : memref<*xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>> loc(#loc6)
    %alloc = memref.alloc() : memref<64x64xf16> loc(#loc6)
    memref.copy %reinterpret_cast, %alloc : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<64x64xf16> loc(#loc6)
    %13 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf16> to tensor<64x64xf16> loc(#loc6)
    %14:5 = scf.for %arg14 = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%arg15 = %c0_i64, %arg16 = %c0_i64, %arg17 = %5, %arg18 = %1, %arg19 = %4) -> (i64, i64, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32>)  : i32 {
      %20 = arith.index_cast %arg16 : i64 to index loc(#loc11)
      %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%20], sizes: [64, 64], strides: [%10, 1] : memref<*xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>> loc(#loc11)
      %alloc_5 = memref.alloc() : memref<64x64xf16> loc(#loc11)
      memref.copy %reinterpret_cast_4, %alloc_5 : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<64x64xf16> loc(#loc11)
      %21 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x64xf16> to tensor<64x64xf16> loc(#loc11)
      %22 = linalg.matmul ins(%13, %21 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32> loc(#loc4)
      %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%22, %3 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%22 : tensor<64x64xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":77:29), %in_12: f32 loc(unknown), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":77:29)):
        %44 = arith.mulf %in, %in_12 : f32 loc(#loc15)
        linalg.yield %44 : f32 loc(#loc15)
      } -> tensor<64x64xf32> loc(#loc15)
      %transposed = linalg.transpose ins(%23 : tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) permutation = [1, 0]  loc(#loc39)
      %reduced = linalg.reduce ins(%transposed : tensor<64x64xf32>) outs(%5 : tensor<64xf32>) dimensions = [0] 
        (%in: f32 loc(callsite(#loc2 at #loc3)), %init: f32 loc(callsite(#loc2 at #loc3))) {
          %44 = arith.maxnumf %in, %init : f32 loc(#loc39)
          linalg.yield %44 : f32 loc(#loc39)
        } loc(#loc39)
      %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg17, %reduced : tensor<64xf32>, tensor<64xf32>) outs(%arg17 : tensor<64xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmp_88odyst/tt.mlir":46:109), %in_12: f32 loc(callsite(#loc2 at #loc3)), %out: f32 loc("/tmp/tmp_88odyst/tt.mlir":46:109)):
        %44 = arith.maxnumf %in, %in_12 : f32 loc(#loc16)
        linalg.yield %44 : f32 loc(#loc16)
      } -> tensor<64xf32> loc(#loc16)
      %expanded_6 = tensor.expand_shape %24 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32> loc(#loc18)
      %25 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_6 : tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":81:23), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":81:18)):
        linalg.yield %in : f32 loc(#loc19)
      } -> tensor<64x64xf32> loc(#loc19)
      %26 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%23, %25 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%23 : tensor<64x64xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":77:40), %in_12: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":81:18), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":77:40)):
        %44 = arith.subf %in, %in_12 : f32 loc(#loc19)
        linalg.yield %44 : f32 loc(#loc19)
      } -> tensor<64x64xf32> loc(#loc19)
      %27 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%26 : tensor<64x64xf32>) outs(%26 : tensor<64x64xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":81:18), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":81:18)):
        %44 = math.exp %in : f32 loc(#loc20)
        linalg.yield %44 : f32 loc(#loc20)
      } -> tensor<64x64xf32> loc(#loc20)
      %transposed_7 = linalg.transpose ins(%27 : tensor<64x64xf32>) outs(%2 : tensor<64x64xf32>) permutation = [1, 0]  loc(#loc45)
      %28 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32> loc(#loc45)
      %reduced_8 = linalg.reduce ins(%transposed_7 : tensor<64x64xf32>) outs(%28 : tensor<64xf32>) dimensions = [0] 
        (%in: f32 loc(callsite(#loc21 at #loc22)), %init: f32 loc(callsite(#loc21 at #loc22))) {
          %44 = arith.addf %in, %init : f32 loc(#loc45)
          linalg.yield %44 : f32 loc(#loc45)
        } loc(#loc45)
      %29 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg17, %24 : tensor<64xf32>, tensor<64xf32>) outs(%arg17 : tensor<64xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmp_88odyst/tt.mlir":46:109), %in_12: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":80:31), %out: f32 loc("/tmp/tmp_88odyst/tt.mlir":46:109)):
        %44 = arith.subf %in, %in_12 : f32 loc(#loc23)
        linalg.yield %44 : f32 loc(#loc23)
      } -> tensor<64xf32> loc(#loc23)
      %30 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%29 : tensor<64xf32>) outs(%29 : tensor<64xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":84:29), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":84:29)):
        %44 = math.exp %in : f32 loc(#loc24)
        linalg.yield %44 : f32 loc(#loc24)
      } -> tensor<64xf32> loc(#loc24)
      %31 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg18, %30 : tensor<64xf32>, tensor<64xf32>) outs(%arg18 : tensor<64xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmp_88odyst/tt.mlir":46:126), %in_12: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":84:23), %out: f32 loc("/tmp/tmp_88odyst/tt.mlir":46:126)):
        %44 = arith.mulf %in, %in_12 : f32 loc(#loc25)
        linalg.yield %44 : f32 loc(#loc25)
      } -> tensor<64xf32> loc(#loc25)
      %32 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%31, %reduced_8 : tensor<64xf32>, tensor<64xf32>) outs(%31 : tensor<64xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":85:20), %in_12: f32 loc(callsite(#loc21 at #loc22)), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":85:20)):
        %44 = arith.addf %in, %in_12 : f32 loc(#loc27)
        linalg.yield %44 : f32 loc(#loc27)
      } -> tensor<64xf32> loc(#loc27)
      %33 = arith.index_cast %arg15 : i64 to index loc(#loc9)
      %34 = arith.muli %33, %9 : index loc(#loc9)
      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [64, 64], strides: [%9, 1] : memref<*xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>> loc(#loc9)
      %alloc_10 = memref.alloc() : memref<64x64xf16> loc(#loc9)
      memref.copy %reinterpret_cast_9, %alloc_10 : memref<64x64xf16, strided<[?, 1], offset: ?>> to memref<64x64xf16> loc(#loc9)
      %35 = bufferization.to_tensor %alloc_10 restrict writable : memref<64x64xf16> to tensor<64x64xf16> loc(#loc9)
      %36 = tensor.empty() : tensor<64x64xf16> loc(#loc28)
      %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<64x64xf32>) outs(%36 : tensor<64x64xf16>) {
      ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":82:19), %out: f16 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":88:17)):
        %44 = arith.truncf %in : f32 to f16 loc(#loc28)
        linalg.yield %44 : f16 loc(#loc28)
      } -> tensor<64x64xf16> loc(#loc28)
      %expanded_11 = tensor.expand_shape %30 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32> loc(#loc29)
      %38 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_11 : tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":89:30), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":89:24)):
        linalg.yield %in : f32 loc(#loc30)
      } -> tensor<64x64xf32> loc(#loc30)
      %39 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg19, %38 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%arg19 : tensor<64x64xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmp_88odyst/tt.mlir":46:141), %in_12: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":89:24), %out: f32 loc("/tmp/tmp_88odyst/tt.mlir":46:141)):
        %44 = arith.mulf %in, %in_12 : f32 loc(#loc30)
        linalg.yield %44 : f32 loc(#loc30)
      } -> tensor<64x64xf32> loc(#loc30)
      %40 = linalg.matmul ins(%37, %35 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%4 : tensor<64x64xf32>) -> tensor<64x64xf32> loc(#loc32)
      %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%39, %40 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%39 : tensor<64x64xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":89:24), %in_12: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":90:35), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":89:24)):
        %44 = arith.addf %in, %in_12 : f32 loc(#loc32)
        linalg.yield %44 : f32 loc(#loc32)
      } -> tensor<64x64xf32> loc(#loc32)
      %42 = arith.addi %arg15, %c64_i64 : i64 loc(#loc33)
      %43 = arith.addi %arg16, %c64_i64 : i64 loc(#loc34)
      scf.yield %42, %43, %24, %32, %41 : i64, i64, tensor<64xf32>, tensor<64xf32>, tensor<64x64xf32> loc(#loc35)
    } loc(#loc14)
    %expanded = tensor.expand_shape %14#3 [[0, 1]] output_shape [64, 1] : tensor<64xf32> into tensor<64x1xf32> loc(#loc36)
    %15 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<64x1xf32>) outs(%2 : tensor<64x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":99:24), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":99:20)):
      linalg.yield %in : f32 loc(#loc37)
    } -> tensor<64x64xf32> loc(#loc37)
    %16 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%14#4, %15 : tensor<64x64xf32>, tensor<64x64xf32>) outs(%14#4 : tensor<64x64xf32>) {
    ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":72:35), %in_4: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":99:20), %out: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":72:35)):
      %20 = arith.divf %in, %in_4 : f32 loc(#loc37)
      linalg.yield %20 : f32 loc(#loc37)
    } -> tensor<64x64xf32> loc(#loc37)
    %17 = tensor.empty() : tensor<64x64xf16> loc(#loc38)
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<64x64xf32>) outs(%17 : tensor<64x64xf16>) {
    ^bb0(%in: f32 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":99:20), %out: f16 loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":100:29)):
      %20 = arith.truncf %in : f32 to f16 loc(#loc38)
      linalg.yield %20 : f16 loc(#loc38)
    } -> tensor<64x64xf16> loc(#loc38)
    %19 = arith.muli %8, %11 : index loc(#loc8)
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%19], sizes: [64, 64], strides: [%11, 1] : memref<*xf16> to memref<64x64xf16, strided<[?, 1], offset: ?>> loc(#loc8)
    bufferization.materialize_in_destination %18 in writable %reinterpret_cast_3 : (tensor<64x64xf16>, memref<64x64xf16, strided<[?, 1], offset: ?>>) -> () loc(#loc8)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc5 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":37:25)
#loc6 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":64:22)
#loc7 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":39:8)
#loc8 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":100:20)
#loc9 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":87:26)
#loc10 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":46:8)
#loc11 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":75:26)
#loc12 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":53:8)
#loc13 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":60:8)
#loc27 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":85:28)
#loc33 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":95:34)
#loc34 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":96:36)
#loc35 = loc("/home/zhenyu/tmd/test/Dialect/Triton/flashattn/flashattn.py":96:8)
#loc40 = loc(fused[#loc6, #loc7])
#loc41 = loc(fused[#loc8, #loc7])
#loc42 = loc(fused[#loc9, #loc10])
#loc43 = loc(fused[#loc11, #loc12])
#loc44 = loc(fused[#loc8, #loc13])

