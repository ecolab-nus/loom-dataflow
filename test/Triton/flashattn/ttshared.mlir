#loc = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0)
#loc1 = loc(unknown)
#loc2 = loc("/home/zhenyu/triton/python/triton/language/standard.py":188:40)
#loc3 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":80:38)
#loc4 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":77:29)
#loc9 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":72:35)
#loc11 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":77:40)
#loc12 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":80:31)
#loc13 = loc("/tmp/tmpu_tcz7ji/tt.mlir":41:108)
#loc14 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":81:23)
#loc15 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":81:18)
#loc16 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":82:19)
#loc17 = loc("/home/zhenyu/triton/python/triton/language/standard.py":290:36)
#loc18 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":83:22)
#loc19 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":84:29)
#loc20 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":84:23)
#loc21 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":85:20)
#loc22 = loc("/tmp/tmpu_tcz7ji/tt.mlir":41:124)
#loc25 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":88:17)
#loc26 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":89:30)
#loc27 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":89:24)
#loc28 = loc("/tmp/tmpu_tcz7ji/tt.mlir":41:140)
#loc29 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":90:35)
#loc33 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":99:24)
#loc34 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":99:20)
#loc35 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":100:29)
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#loc36 = loc(callsite(#loc2 at #loc3))
#loc38 = loc(callsite(#loc17 at #loc18))
module {
  func.func @flashattn_fwd(%arg0: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg1: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg2: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg3: memref<*xf16> {tt.divisibility = 16 : i32} loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg4: i32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg5: i32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg6: i32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg7: i32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg8: i32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0), %arg9: i32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":10:0)) {
    %cst = arith.constant 1.000000e+00 : f32 loc(#loc1)
    %c0_i64 = arith.constant 0 : i64 loc(#loc1)
    %c32_i64 = arith.constant 32 : i64 loc(#loc1)
    %c32 = arith.constant 32 : index loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c512_i32 = arith.constant 512 : i32 loc(#loc1)
    %cst_0 = arith.constant 0.176776692 : f32 loc(#loc1)
    %cst_1 = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %cst_2 = arith.constant 0xFF800000 : f32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %0 = tensor.empty() : tensor<32xf32> loc(#loc36)
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32> loc(#loc1)
    %2 = tensor.empty() : tensor<32x32xf32> loc(#loc4)
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32> loc(#loc1)
    %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32> loc(#loc4)
    %5 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32> loc(#loc36)
    %6 = arith.muli %arg7, %c32_i32 : i32 loc(#loc5)
    %7 = arith.index_cast %6 : i32 to index loc(#loc37)
    %8 = arith.muli %7, %c32 : index loc(#loc8)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%8], sizes: [32, 32], strides: [32, 1] : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>> loc(#loc8)
    %alloc = memref.alloc() : memref<32x32xf16> loc(#loc8)
    memref.copy %reinterpret_cast, %alloc : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16> loc(#loc8)
    %9 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16> to tensor<32x32xf16> loc(#loc8)
    %10:5 = scf.for %arg10 = %c0_i32 to %c512_i32 step %c32_i32 iter_args(%arg11 = %c0_i64, %arg12 = %c0_i64, %arg13 = %5, %arg14 = %1, %arg15 = %4) -> (i64, i64, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>)  : i32 {
      %15 = arith.index_cast %arg12 : i64 to index loc(#loc10)
      %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%15], sizes: [32, 32], strides: [512, 1] : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>> loc(#loc10)
      %alloc_5 = memref.alloc() : memref<32x32xf16> loc(#loc10)
      memref.copy %reinterpret_cast_4, %alloc_5 : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16> loc(#loc10)
      %16 = bufferization.to_tensor %alloc_5 restrict writable : memref<32x32xf16> to tensor<32x32xf16> loc(#loc10)
      %17 = linalg.matmul ins(%9, %16 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32> loc(#loc4)
      %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%17, %3 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%17 : tensor<32x32xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":77:29), %in_12: f32 loc(unknown), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":77:29)):
        %39 = arith.mulf %in, %in_12 : f32 loc(#loc11)
        linalg.yield %39 : f32 loc(#loc11)
      } -> tensor<32x32xf32> loc(#loc11)
      %transposed = linalg.transpose ins(%18 : tensor<32x32xf32>) outs(%2 : tensor<32x32xf32>) permutation = [1, 0]  loc(#loc36)
      %reduced = linalg.reduce ins(%transposed : tensor<32x32xf32>) outs(%5 : tensor<32xf32>) dimensions = [0] 
        (%in: f32 loc(callsite(#loc2 at #loc3)), %init: f32 loc(callsite(#loc2 at #loc3))) {
          %39 = arith.maxnumf %in, %init : f32 loc(#loc36)
          linalg.yield %39 : f32 loc(#loc36)
        } loc(#loc36)
      %19 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg13, %reduced : tensor<32xf32>, tensor<32xf32>) outs(%arg13 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpu_tcz7ji/tt.mlir":41:108), %in_12: f32 loc(callsite(#loc2 at #loc3)), %out: f32 loc("/tmp/tmpu_tcz7ji/tt.mlir":41:108)):
        %39 = arith.maxnumf %in, %in_12 : f32 loc(#loc12)
        linalg.yield %39 : f32 loc(#loc12)
      } -> tensor<32xf32> loc(#loc12)
      %expanded_6 = tensor.expand_shape %19 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32> loc(#loc14)
      %20 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_6 : tensor<32x1xf32>) outs(%2 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":81:23), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":81:18)):
        linalg.yield %in : f32 loc(#loc15)
      } -> tensor<32x32xf32> loc(#loc15)
      %21 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%18, %20 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%18 : tensor<32x32xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":77:40), %in_12: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":81:18), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":77:40)):
        %39 = arith.subf %in, %in_12 : f32 loc(#loc15)
        linalg.yield %39 : f32 loc(#loc15)
      } -> tensor<32x32xf32> loc(#loc15)
      %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%21 : tensor<32x32xf32>) outs(%21 : tensor<32x32xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":81:18), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":81:18)):
        %39 = math.exp %in : f32 loc(#loc16)
        linalg.yield %39 : f32 loc(#loc16)
      } -> tensor<32x32xf32> loc(#loc16)
      %transposed_7 = linalg.transpose ins(%22 : tensor<32x32xf32>) outs(%2 : tensor<32x32xf32>) permutation = [1, 0]  loc(#loc38)
      %23 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32> loc(#loc38)
      %reduced_8 = linalg.reduce ins(%transposed_7 : tensor<32x32xf32>) outs(%23 : tensor<32xf32>) dimensions = [0] 
        (%in: f32 loc(callsite(#loc17 at #loc18)), %init: f32 loc(callsite(#loc17 at #loc18))) {
          %39 = arith.addf %in, %init : f32 loc(#loc38)
          linalg.yield %39 : f32 loc(#loc38)
        } loc(#loc38)
      %24 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg13, %19 : tensor<32xf32>, tensor<32xf32>) outs(%arg13 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpu_tcz7ji/tt.mlir":41:108), %in_12: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":80:31), %out: f32 loc("/tmp/tmpu_tcz7ji/tt.mlir":41:108)):
        %39 = arith.subf %in, %in_12 : f32 loc(#loc19)
        linalg.yield %39 : f32 loc(#loc19)
      } -> tensor<32xf32> loc(#loc19)
      %25 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%24 : tensor<32xf32>) outs(%24 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":84:29), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":84:29)):
        %39 = math.exp %in : f32 loc(#loc20)
        linalg.yield %39 : f32 loc(#loc20)
      } -> tensor<32xf32> loc(#loc20)
      %26 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg14, %25 : tensor<32xf32>, tensor<32xf32>) outs(%arg14 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpu_tcz7ji/tt.mlir":41:124), %in_12: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":84:23), %out: f32 loc("/tmp/tmpu_tcz7ji/tt.mlir":41:124)):
        %39 = arith.mulf %in, %in_12 : f32 loc(#loc21)
        linalg.yield %39 : f32 loc(#loc21)
      } -> tensor<32xf32> loc(#loc21)
      %27 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%26, %reduced_8 : tensor<32xf32>, tensor<32xf32>) outs(%26 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":85:20), %in_12: f32 loc(callsite(#loc17 at #loc18)), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":85:20)):
        %39 = arith.addf %in, %in_12 : f32 loc(#loc23)
        linalg.yield %39 : f32 loc(#loc23)
      } -> tensor<32xf32> loc(#loc23)
      %28 = arith.index_cast %arg11 : i64 to index loc(#loc24)
      %29 = arith.muli %28, %c32 : index loc(#loc24)
      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [32, 32], strides: [32, 1] : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>> loc(#loc24)
      %alloc_10 = memref.alloc() : memref<32x32xf16> loc(#loc24)
      memref.copy %reinterpret_cast_9, %alloc_10 : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16> loc(#loc24)
      %30 = bufferization.to_tensor %alloc_10 restrict writable : memref<32x32xf16> to tensor<32x32xf16> loc(#loc24)
      %31 = tensor.empty() : tensor<32x32xf16> loc(#loc25)
      %32 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%22 : tensor<32x32xf32>) outs(%31 : tensor<32x32xf16>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":82:19), %out: f16 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":88:17)):
        %39 = arith.truncf %in : f32 to f16 loc(#loc25)
        linalg.yield %39 : f16 loc(#loc25)
      } -> tensor<32x32xf16> loc(#loc25)
      %expanded_11 = tensor.expand_shape %25 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32> loc(#loc26)
      %33 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_11 : tensor<32x1xf32>) outs(%2 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":89:30), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":89:24)):
        linalg.yield %in : f32 loc(#loc27)
      } -> tensor<32x32xf32> loc(#loc27)
      %34 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg15, %33 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg15 : tensor<32x32xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpu_tcz7ji/tt.mlir":41:140), %in_12: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":89:24), %out: f32 loc("/tmp/tmpu_tcz7ji/tt.mlir":41:140)):
        %39 = arith.mulf %in, %in_12 : f32 loc(#loc27)
        linalg.yield %39 : f32 loc(#loc27)
      } -> tensor<32x32xf32> loc(#loc27)
      %35 = linalg.matmul ins(%32, %30 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32> loc(#loc29)
      %36 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%34, %35 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%34 : tensor<32x32xf32>) {
      ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":89:24), %in_12: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":90:35), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":89:24)):
        %39 = arith.addf %in, %in_12 : f32 loc(#loc29)
        linalg.yield %39 : f32 loc(#loc29)
      } -> tensor<32x32xf32> loc(#loc29)
      %37 = arith.addi %arg11, %c32_i64 : i64 loc(#loc30)
      %38 = arith.addi %arg12, %c32_i64 : i64 loc(#loc31)
      scf.yield %37, %38, %19, %27, %36 : i64, i64, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32> loc(#loc32)
    } loc(#loc9)
    %expanded = tensor.expand_shape %10#3 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32> loc(#loc33)
    %11 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<32x1xf32>) outs(%2 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":99:24), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":99:20)):
      linalg.yield %in : f32 loc(#loc34)
    } -> tensor<32x32xf32> loc(#loc34)
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10#4, %11 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%10#4 : tensor<32x32xf32>) {
    ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":72:35), %in_4: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":99:20), %out: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":72:35)):
      %15 = arith.divf %in, %in_4 : f32 loc(#loc34)
      linalg.yield %15 : f32 loc(#loc34)
    } -> tensor<32x32xf32> loc(#loc34)
    %13 = tensor.empty() : tensor<32x32xf16> loc(#loc35)
    %14 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%12 : tensor<32x32xf32>) outs(%13 : tensor<32x32xf16>) {
    ^bb0(%in: f32 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":99:20), %out: f16 loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":100:29)):
      %15 = arith.truncf %in : f32 to f16 loc(#loc35)
      linalg.yield %15 : f16 loc(#loc35)
    } -> tensor<32x32xf16> loc(#loc35)
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%8], sizes: [32, 32], strides: [32, 1] : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>> loc(#loc6)
    bufferization.materialize_in_destination %14 in writable %reinterpret_cast_3 : (tensor<32x32xf16>, memref<32x32xf16, strided<[32, 1], offset: ?>>) -> () loc(#loc6)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc5 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":37:25)
#loc6 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":100:20)
#loc7 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":39:8)
#loc8 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":64:22)
#loc10 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":75:26)
#loc23 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":85:28)
#loc24 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":87:26)
#loc30 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":95:34)
#loc31 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":96:36)
#loc32 = loc("/home/zhenyu/loom/test/Dialect/Triton/flashattn/flashattn.py":96:8)
#loc37 = loc(fused[#loc6, #loc7])

