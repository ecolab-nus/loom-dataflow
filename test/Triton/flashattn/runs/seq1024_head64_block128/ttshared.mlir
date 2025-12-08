#loc = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0)
#loc1 = loc(unknown)
#loc2 = loc("/home/zhenyu/triton/python/triton/language/standard.py":188:40)
#loc3 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":92:38)
#loc4 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":101:24)
#loc5 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":102:35)
#loc6 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":89:29)
#loc11 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":84:35)
#loc13 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":89:40)
#loc14 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":92:31)
#loc15 = loc("/tmp/tmpvydew0la/tt.mlir":44:110)
#loc16 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":93:23)
#loc17 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":93:18)
#loc18 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":94:19)
#loc19 = loc("/home/zhenyu/triton/python/triton/language/standard.py":290:36)
#loc20 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":95:22)
#loc21 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":96:29)
#loc22 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":96:23)
#loc23 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":97:20)
#loc24 = loc("/tmp/tmpvydew0la/tt.mlir":44:126)
#loc27 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":100:17)
#loc28 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":101:30)
#loc29 = loc("/tmp/tmpvydew0la/tt.mlir":44:142)
#loc33 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":111:24)
#loc34 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":111:20)
#loc35 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":112:29)
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#loc36 = loc(callsite(#loc2 at #loc3))
#loc38 = loc(callsite(#loc19 at #loc20))
module {
  func.func @flashattn_fwd(%arg0: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0), %arg1: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0), %arg2: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0), %arg3: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0), %arg4: i32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0), %arg5: i32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0), %arg6: i32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0), %arg7: i32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0), %arg8: i32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0), %arg9: i32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":20:0)) {
    %cst = arith.constant 1.000000e+00 : f32 loc(#loc1)
    %c0_i64 = arith.constant 0 : i64 loc(#loc1)
    %c128_i64 = arith.constant 128 : i64 loc(#loc1)
    %c64 = arith.constant 64 : index loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %cst_0 = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %cst_1 = arith.constant 1.250000e-01 : f32 loc(#loc1)
    %cst_2 = arith.constant 0xFF800000 : f32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %0 = tensor.empty() : tensor<128xf32> loc(#loc36)
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128xf32>) -> tensor<128xf32> loc(#loc1)
    %2 = tensor.empty() : tensor<128x64xf32> loc(#loc4)
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<128x64xf32>) -> tensor<128x64xf32> loc(#loc5)
    %4 = tensor.empty() : tensor<128x128xf32> loc(#loc6)
    %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<128x128xf32>) -> tensor<128x128xf32> loc(#loc1)
    %6 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<128xf32>) -> tensor<128xf32> loc(#loc36)
    %7 = arith.muli %arg7, %c128_i32 : i32 loc(#loc7)
    %8 = arith.index_cast %7 : i32 to index loc(#loc37)
    %9 = arith.muli %8, %c64 : index loc(#loc10)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [128, 64], strides: [64, 1] : memref<*xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>> loc(#loc10)
    %alloc = memref.alloc() : memref<128x64xf16> loc(#loc10)
    memref.copy %reinterpret_cast, %alloc : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16> loc(#loc10)
    %10 = bufferization.to_tensor %alloc restrict writable : memref<128x64xf16> to tensor<128x64xf16> loc(#loc10)
    %11:5 = scf.for %arg10 = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%arg11 = %c0_i64, %arg12 = %c0_i64, %arg13 = %6, %arg14 = %1, %arg15 = %3) -> (i64, i64, tensor<128xf32>, tensor<128xf32>, tensor<128x64xf32>)  : i32 {
      %16 = arith.index_cast %arg12 : i64 to index loc(#loc12)
      %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%16], sizes: [64, 128], strides: [1024, 1] : memref<*xf16> to memref<64x128xf16, strided<[1024, 1], offset: ?>> loc(#loc12)
      %alloc_5 = memref.alloc() : memref<64x128xf16> loc(#loc12)
      memref.copy %reinterpret_cast_4, %alloc_5 : memref<64x128xf16, strided<[1024, 1], offset: ?>> to memref<64x128xf16> loc(#loc12)
      %17 = bufferization.to_tensor %alloc_5 restrict writable : memref<64x128xf16> to tensor<64x128xf16> loc(#loc12)
      %18 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<128x128xf32>) -> tensor<128x128xf32> loc(#loc6)
      %19 = linalg.matmul ins(%10, %17 : tensor<128x64xf16>, tensor<64x128xf16>) outs(%18 : tensor<128x128xf32>) -> tensor<128x128xf32> loc(#loc6)
      %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%19, %5 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%19 : tensor<128x128xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":89:29), %in_12: f32 loc(unknown), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":89:29)):
        %41 = arith.mulf %in, %in_12 : f32 loc(#loc13)
        linalg.yield %41 : f32 loc(#loc13)
      } -> tensor<128x128xf32> loc(#loc13)
      %transposed = linalg.transpose ins(%20 : tensor<128x128xf32>) outs(%4 : tensor<128x128xf32>) permutation = [1, 0]  loc(#loc36)
      %reduced = linalg.reduce ins(%transposed : tensor<128x128xf32>) outs(%6 : tensor<128xf32>) dimensions = [0] 
        (%in: f32 loc(callsite(#loc2 at #loc3)), %init: f32 loc(callsite(#loc2 at #loc3))) {
          %41 = arith.maxnumf %in, %init : f32 loc(#loc36)
          linalg.yield %41 : f32 loc(#loc36)
        } loc(#loc36)
      %21 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg13, %reduced : tensor<128xf32>, tensor<128xf32>) outs(%arg13 : tensor<128xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpvydew0la/tt.mlir":44:110), %in_12: f32 loc(callsite(#loc2 at #loc3)), %out: f32 loc("/tmp/tmpvydew0la/tt.mlir":44:110)):
        %41 = arith.maxnumf %in, %in_12 : f32 loc(#loc14)
        linalg.yield %41 : f32 loc(#loc14)
      } -> tensor<128xf32> loc(#loc14)
      %expanded_6 = tensor.expand_shape %21 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32> loc(#loc16)
      %22 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_6 : tensor<128x1xf32>) outs(%4 : tensor<128x128xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":93:23), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":93:18)):
        linalg.yield %in : f32 loc(#loc17)
      } -> tensor<128x128xf32> loc(#loc17)
      %23 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%20, %22 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%20 : tensor<128x128xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":89:40), %in_12: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":93:18), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":89:40)):
        %41 = arith.subf %in, %in_12 : f32 loc(#loc17)
        linalg.yield %41 : f32 loc(#loc17)
      } -> tensor<128x128xf32> loc(#loc17)
      %24 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%23 : tensor<128x128xf32>) outs(%23 : tensor<128x128xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":93:18), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":93:18)):
        %41 = math.exp %in : f32 loc(#loc18)
        linalg.yield %41 : f32 loc(#loc18)
      } -> tensor<128x128xf32> loc(#loc18)
      %transposed_7 = linalg.transpose ins(%24 : tensor<128x128xf32>) outs(%4 : tensor<128x128xf32>) permutation = [1, 0]  loc(#loc38)
      %25 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<128xf32>) -> tensor<128xf32> loc(#loc38)
      %reduced_8 = linalg.reduce ins(%transposed_7 : tensor<128x128xf32>) outs(%25 : tensor<128xf32>) dimensions = [0] 
        (%in: f32 loc(callsite(#loc19 at #loc20)), %init: f32 loc(callsite(#loc19 at #loc20))) {
          %41 = arith.addf %in, %init : f32 loc(#loc38)
          linalg.yield %41 : f32 loc(#loc38)
        } loc(#loc38)
      %26 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg13, %21 : tensor<128xf32>, tensor<128xf32>) outs(%arg13 : tensor<128xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpvydew0la/tt.mlir":44:110), %in_12: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":92:31), %out: f32 loc("/tmp/tmpvydew0la/tt.mlir":44:110)):
        %41 = arith.subf %in, %in_12 : f32 loc(#loc21)
        linalg.yield %41 : f32 loc(#loc21)
      } -> tensor<128xf32> loc(#loc21)
      %27 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%26 : tensor<128xf32>) outs(%26 : tensor<128xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":96:29), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":96:29)):
        %41 = math.exp %in : f32 loc(#loc22)
        linalg.yield %41 : f32 loc(#loc22)
      } -> tensor<128xf32> loc(#loc22)
      %28 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg14, %27 : tensor<128xf32>, tensor<128xf32>) outs(%arg14 : tensor<128xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpvydew0la/tt.mlir":44:126), %in_12: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":96:23), %out: f32 loc("/tmp/tmpvydew0la/tt.mlir":44:126)):
        %41 = arith.mulf %in, %in_12 : f32 loc(#loc23)
        linalg.yield %41 : f32 loc(#loc23)
      } -> tensor<128xf32> loc(#loc23)
      %29 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%28, %reduced_8 : tensor<128xf32>, tensor<128xf32>) outs(%28 : tensor<128xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":97:20), %in_12: f32 loc(callsite(#loc19 at #loc20)), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":97:20)):
        %41 = arith.addf %in, %in_12 : f32 loc(#loc25)
        linalg.yield %41 : f32 loc(#loc25)
      } -> tensor<128xf32> loc(#loc25)
      %30 = arith.index_cast %arg11 : i64 to index loc(#loc26)
      %31 = arith.muli %30, %c64 : index loc(#loc26)
      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [128, 64], strides: [64, 1] : memref<*xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>> loc(#loc26)
      %alloc_10 = memref.alloc() : memref<128x64xf16> loc(#loc26)
      memref.copy %reinterpret_cast_9, %alloc_10 : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<128x64xf16> loc(#loc26)
      %32 = bufferization.to_tensor %alloc_10 restrict writable : memref<128x64xf16> to tensor<128x64xf16> loc(#loc26)
      %33 = tensor.empty() : tensor<128x128xf16> loc(#loc27)
      %34 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%24 : tensor<128x128xf32>) outs(%33 : tensor<128x128xf16>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":94:19), %out: f16 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":100:17)):
        %41 = arith.truncf %in : f32 to f16 loc(#loc27)
        linalg.yield %41 : f16 loc(#loc27)
      } -> tensor<128x128xf16> loc(#loc27)
      %expanded_11 = tensor.expand_shape %27 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32> loc(#loc28)
      %35 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_11 : tensor<128x1xf32>) outs(%2 : tensor<128x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":101:30), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":101:24)):
        linalg.yield %in : f32 loc(#loc4)
      } -> tensor<128x64xf32> loc(#loc4)
      %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg15, %35 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%arg15 : tensor<128x64xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpvydew0la/tt.mlir":44:142), %in_12: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":101:24), %out: f32 loc("/tmp/tmpvydew0la/tt.mlir":44:142)):
        %41 = arith.mulf %in, %in_12 : f32 loc(#loc4)
        linalg.yield %41 : f32 loc(#loc4)
      } -> tensor<128x64xf32> loc(#loc4)
      %37 = linalg.matmul ins(%34, %32 : tensor<128x128xf16>, tensor<128x64xf16>) outs(%3 : tensor<128x64xf32>) -> tensor<128x64xf32> loc(#loc5)
      %38 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%36, %37 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%36 : tensor<128x64xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":101:24), %in_12: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":102:35), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":101:24)):
        %41 = arith.addf %in, %in_12 : f32 loc(#loc5)
        linalg.yield %41 : f32 loc(#loc5)
      } -> tensor<128x64xf32> loc(#loc5)
      %39 = arith.addi %arg11, %c128_i64 : i64 loc(#loc30)
      %40 = arith.addi %arg12, %c128_i64 : i64 loc(#loc31)
      scf.yield %39, %40, %21, %29, %38 : i64, i64, tensor<128xf32>, tensor<128xf32>, tensor<128x64xf32> loc(#loc32)
    } loc(#loc11)
    %expanded = tensor.expand_shape %11#3 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32> loc(#loc33)
    %12 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<128x1xf32>) outs(%2 : tensor<128x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":111:24), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":111:20)):
      linalg.yield %in : f32 loc(#loc34)
    } -> tensor<128x64xf32> loc(#loc34)
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11#4, %12 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%11#4 : tensor<128x64xf32>) {
    ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":84:35), %in_4: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":111:20), %out: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":84:35)):
      %16 = arith.divf %in, %in_4 : f32 loc(#loc34)
      linalg.yield %16 : f32 loc(#loc34)
    } -> tensor<128x64xf32> loc(#loc34)
    %14 = tensor.empty() : tensor<128x64xf16> loc(#loc35)
    %15 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<128x64xf32>) outs(%14 : tensor<128x64xf16>) {
    ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":111:20), %out: f16 loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":112:29)):
      %16 = arith.truncf %in : f32 to f16 loc(#loc35)
      linalg.yield %16 : f16 loc(#loc35)
    } -> tensor<128x64xf16> loc(#loc35)
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [128, 64], strides: [64, 1] : memref<*xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>> loc(#loc8)
    bufferization.materialize_in_destination %15 in writable %reinterpret_cast_3 : (tensor<128x64xf16>, memref<128x64xf16, strided<[64, 1], offset: ?>>) -> () loc(#loc8)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc7 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":49:25)
#loc8 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":112:20)
#loc9 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":51:8)
#loc10 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":76:22)
#loc12 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":87:26)
#loc25 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":97:28)
#loc26 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":99:26)
#loc30 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":107:34)
#loc31 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":108:36)
#loc32 = loc("/home/zhenyu/loom/test/Triton/flashattn/flashattn.py":108:8)
#loc37 = loc(fused[#loc8, #loc9])

