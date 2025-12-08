#loc = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0)
#loc1 = loc(unknown)
#loc2 = loc("/path/to/triton/python/triton/language/standard.py":188:40)
#loc3 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":92:38)
#loc4 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":101:24)
#loc5 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":102:35)
#loc6 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":89:29)
#loc11 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":84:35)
#loc13 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":89:40)
#loc14 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":92:31)
#loc15 = loc("/tmp/tmpsip4oz60/tt.mlir":47:108)
#loc16 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":93:23)
#loc17 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":93:18)
#loc18 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":94:19)
#loc19 = loc("/path/to/triton/python/triton/language/standard.py":290:36)
#loc20 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":95:22)
#loc21 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":96:29)
#loc22 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":96:23)
#loc23 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":97:20)
#loc24 = loc("/tmp/tmpsip4oz60/tt.mlir":47:124)
#loc27 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":100:17)
#loc28 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":101:30)
#loc29 = loc("/tmp/tmpsip4oz60/tt.mlir":47:140)
#loc33 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":111:24)
#loc34 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":111:20)
#loc35 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":112:29)
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
#loc36 = loc(callsite(#loc2 at #loc3))
#loc38 = loc(callsite(#loc19 at #loc20))
module {
  func.func @flashattn_fwd(%arg0: memref<*xf16> {tt.divisibility = 16 : i32} loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0), %arg1: memref<*xf16> {tt.divisibility = 16 : i32} loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0), %arg2: memref<*xf16> {tt.divisibility = 16 : i32} loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0), %arg3: memref<*xf16> {tt.divisibility = 16 : i32} loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0), %arg4: i32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0), %arg5: i32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0), %arg6: i32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0), %arg7: i32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0), %arg8: i32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0), %arg9: i32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":20:0)) {
    %cst = arith.constant 1.000000e+00 : f32 loc(#loc1)
    %c0_i64 = arith.constant 0 : i64 loc(#loc1)
    %c64_i64 = arith.constant 64 : i64 loc(#loc1)
    %c32 = arith.constant 32 : index loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %cst_0 = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %cst_1 = arith.constant 0.176776692 : f32 loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %cst_2 = arith.constant 0xFF800000 : f32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %0 = tensor.empty() : tensor<32xf32> loc(#loc36)
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32> loc(#loc1)
    %2 = tensor.empty() : tensor<32x32xf32> loc(#loc4)
    %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32> loc(#loc5)
    %4 = tensor.empty() : tensor<32x64xf32> loc(#loc6)
    %5 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<32x64xf32>) -> tensor<32x64xf32> loc(#loc1)
    %6 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32> loc(#loc36)
    %7 = arith.muli %arg7, %c32_i32 : i32 loc(#loc7)
    %8 = arith.index_cast %7 : i32 to index loc(#loc37)
    %9 = arith.muli %8, %c32 : index loc(#loc10)
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%9], sizes: [32, 32], strides: [32, 1] : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>> loc(#loc10)
    %alloc = memref.alloc() : memref<32x32xf16> loc(#loc10)
    memref.copy %reinterpret_cast, %alloc : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16> loc(#loc10)
    %10 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16> to tensor<32x32xf16> loc(#loc10)
    %11:5 = scf.for %arg10 = %c0_i32 to %c256_i32 step %c64_i32 iter_args(%arg11 = %c0_i64, %arg12 = %c0_i64, %arg13 = %6, %arg14 = %1, %arg15 = %3) -> (i64, i64, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>)  : i32 {
      %16 = arith.index_cast %arg12 : i64 to index loc(#loc12)
      %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%16], sizes: [32, 64], strides: [256, 1] : memref<*xf16> to memref<32x64xf16, strided<[256, 1], offset: ?>> loc(#loc12)
      %alloc_5 = memref.alloc() : memref<32x64xf16> loc(#loc12)
      memref.copy %reinterpret_cast_4, %alloc_5 : memref<32x64xf16, strided<[256, 1], offset: ?>> to memref<32x64xf16> loc(#loc12)
      %17 = bufferization.to_tensor %alloc_5 restrict writable : memref<32x64xf16> to tensor<32x64xf16> loc(#loc12)
      %18 = linalg.fill ins(%cst_0 : f32) outs(%4 : tensor<32x64xf32>) -> tensor<32x64xf32> loc(#loc6)
      %19 = linalg.matmul ins(%10, %17 : tensor<32x32xf16>, tensor<32x64xf16>) outs(%18 : tensor<32x64xf32>) -> tensor<32x64xf32> loc(#loc6)
      %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%19, %5 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%19 : tensor<32x64xf32>) {
      ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":89:29), %in_12: f32 loc(unknown), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":89:29)):
        %42 = arith.mulf %in, %in_12 : f32 loc(#loc13)
        linalg.yield %42 : f32 loc(#loc13)
      } -> tensor<32x64xf32> loc(#loc13)
      %21 = tensor.empty() : tensor<64x32xf32> loc(#loc36)
      %transposed = linalg.transpose ins(%20 : tensor<32x64xf32>) outs(%21 : tensor<64x32xf32>) permutation = [1, 0]  loc(#loc36)
      %reduced = linalg.reduce ins(%transposed : tensor<64x32xf32>) outs(%6 : tensor<32xf32>) dimensions = [0] 
        (%in: f32 loc(callsite(#loc2 at #loc3)), %init: f32 loc(callsite(#loc2 at #loc3))) {
          %42 = arith.maxnumf %in, %init : f32 loc(#loc36)
          linalg.yield %42 : f32 loc(#loc36)
        } loc(#loc36)
      %22 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg13, %reduced : tensor<32xf32>, tensor<32xf32>) outs(%arg13 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpsip4oz60/tt.mlir":47:108), %in_12: f32 loc(callsite(#loc2 at #loc3)), %out: f32 loc("/tmp/tmpsip4oz60/tt.mlir":47:108)):
        %42 = arith.maxnumf %in, %in_12 : f32 loc(#loc14)
        linalg.yield %42 : f32 loc(#loc14)
      } -> tensor<32xf32> loc(#loc14)
      %expanded_6 = tensor.expand_shape %22 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32> loc(#loc16)
      %23 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_6 : tensor<32x1xf32>) outs(%4 : tensor<32x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":93:23), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":93:18)):
        linalg.yield %in : f32 loc(#loc17)
      } -> tensor<32x64xf32> loc(#loc17)
      %24 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%20, %23 : tensor<32x64xf32>, tensor<32x64xf32>) outs(%20 : tensor<32x64xf32>) {
      ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":89:40), %in_12: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":93:18), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":89:40)):
        %42 = arith.subf %in, %in_12 : f32 loc(#loc17)
        linalg.yield %42 : f32 loc(#loc17)
      } -> tensor<32x64xf32> loc(#loc17)
      %25 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%24 : tensor<32x64xf32>) outs(%24 : tensor<32x64xf32>) {
      ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":93:18), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":93:18)):
        %42 = math.exp %in : f32 loc(#loc18)
        linalg.yield %42 : f32 loc(#loc18)
      } -> tensor<32x64xf32> loc(#loc18)
      %transposed_7 = linalg.transpose ins(%25 : tensor<32x64xf32>) outs(%21 : tensor<64x32xf32>) permutation = [1, 0]  loc(#loc38)
      %26 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32> loc(#loc38)
      %reduced_8 = linalg.reduce ins(%transposed_7 : tensor<64x32xf32>) outs(%26 : tensor<32xf32>) dimensions = [0] 
        (%in: f32 loc(callsite(#loc19 at #loc20)), %init: f32 loc(callsite(#loc19 at #loc20))) {
          %42 = arith.addf %in, %init : f32 loc(#loc38)
          linalg.yield %42 : f32 loc(#loc38)
        } loc(#loc38)
      %27 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg13, %22 : tensor<32xf32>, tensor<32xf32>) outs(%arg13 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpsip4oz60/tt.mlir":47:108), %in_12: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":92:31), %out: f32 loc("/tmp/tmpsip4oz60/tt.mlir":47:108)):
        %42 = arith.subf %in, %in_12 : f32 loc(#loc21)
        linalg.yield %42 : f32 loc(#loc21)
      } -> tensor<32xf32> loc(#loc21)
      %28 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel"]} ins(%27 : tensor<32xf32>) outs(%27 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":96:29), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":96:29)):
        %42 = math.exp %in : f32 loc(#loc22)
        linalg.yield %42 : f32 loc(#loc22)
      } -> tensor<32xf32> loc(#loc22)
      %29 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg14, %28 : tensor<32xf32>, tensor<32xf32>) outs(%arg14 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpsip4oz60/tt.mlir":47:124), %in_12: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":96:23), %out: f32 loc("/tmp/tmpsip4oz60/tt.mlir":47:124)):
        %42 = arith.mulf %in, %in_12 : f32 loc(#loc23)
        linalg.yield %42 : f32 loc(#loc23)
      } -> tensor<32xf32> loc(#loc23)
      %30 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%29, %reduced_8 : tensor<32xf32>, tensor<32xf32>) outs(%29 : tensor<32xf32>) {
      ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":97:20), %in_12: f32 loc(callsite(#loc19 at #loc20)), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":97:20)):
        %42 = arith.addf %in, %in_12 : f32 loc(#loc25)
        linalg.yield %42 : f32 loc(#loc25)
      } -> tensor<32xf32> loc(#loc25)
      %31 = arith.index_cast %arg11 : i64 to index loc(#loc26)
      %32 = arith.muli %31, %c32 : index loc(#loc26)
      %reinterpret_cast_9 = memref.reinterpret_cast %arg2 to offset: [%32], sizes: [64, 32], strides: [32, 1] : memref<*xf16> to memref<64x32xf16, strided<[32, 1], offset: ?>> loc(#loc26)
      %alloc_10 = memref.alloc() : memref<64x32xf16> loc(#loc26)
      memref.copy %reinterpret_cast_9, %alloc_10 : memref<64x32xf16, strided<[32, 1], offset: ?>> to memref<64x32xf16> loc(#loc26)
      %33 = bufferization.to_tensor %alloc_10 restrict writable : memref<64x32xf16> to tensor<64x32xf16> loc(#loc26)
      %34 = tensor.empty() : tensor<32x64xf16> loc(#loc27)
      %35 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%25 : tensor<32x64xf32>) outs(%34 : tensor<32x64xf16>) {
      ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":94:19), %out: f16 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":100:17)):
        %42 = arith.truncf %in : f32 to f16 loc(#loc27)
        linalg.yield %42 : f16 loc(#loc27)
      } -> tensor<32x64xf16> loc(#loc27)
      %expanded_11 = tensor.expand_shape %28 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32> loc(#loc28)
      %36 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_11 : tensor<32x1xf32>) outs(%2 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":101:30), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":101:24)):
        linalg.yield %in : f32 loc(#loc4)
      } -> tensor<32x32xf32> loc(#loc4)
      %37 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg15, %36 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg15 : tensor<32x32xf32>) {
      ^bb0(%in: f32 loc("/tmp/tmpsip4oz60/tt.mlir":47:140), %in_12: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":101:24), %out: f32 loc("/tmp/tmpsip4oz60/tt.mlir":47:140)):
        %42 = arith.mulf %in, %in_12 : f32 loc(#loc4)
        linalg.yield %42 : f32 loc(#loc4)
      } -> tensor<32x32xf32> loc(#loc4)
      %38 = linalg.matmul ins(%35, %33 : tensor<32x64xf16>, tensor<64x32xf16>) outs(%3 : tensor<32x32xf32>) -> tensor<32x32xf32> loc(#loc5)
      %39 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%37, %38 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%37 : tensor<32x32xf32>) {
      ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":101:24), %in_12: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":102:35), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":101:24)):
        %42 = arith.addf %in, %in_12 : f32 loc(#loc5)
        linalg.yield %42 : f32 loc(#loc5)
      } -> tensor<32x32xf32> loc(#loc5)
      %40 = arith.addi %arg11, %c64_i64 : i64 loc(#loc30)
      %41 = arith.addi %arg12, %c64_i64 : i64 loc(#loc31)
      scf.yield %40, %41, %22, %30, %39 : i64, i64, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32> loc(#loc32)
    } loc(#loc11)
    %expanded = tensor.expand_shape %11#3 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32> loc(#loc33)
    %12 = linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<32x1xf32>) outs(%2 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":111:24), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":111:20)):
      linalg.yield %in : f32 loc(#loc34)
    } -> tensor<32x32xf32> loc(#loc34)
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%11#4, %12 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%11#4 : tensor<32x32xf32>) {
    ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":84:35), %in_4: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":111:20), %out: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":84:35)):
      %16 = arith.divf %in, %in_4 : f32 loc(#loc34)
      linalg.yield %16 : f32 loc(#loc34)
    } -> tensor<32x32xf32> loc(#loc34)
    %14 = tensor.empty() : tensor<32x32xf16> loc(#loc35)
    %15 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<32x32xf32>) outs(%14 : tensor<32x32xf16>) {
    ^bb0(%in: f32 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":111:20), %out: f16 loc("/path/to/loom/test/Triton/flashattn/flashattn.py":112:29)):
      %16 = arith.truncf %in : f32 to f16 loc(#loc35)
      linalg.yield %16 : f16 loc(#loc35)
    } -> tensor<32x32xf16> loc(#loc35)
    %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%9], sizes: [32, 32], strides: [32, 1] : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>> loc(#loc8)
    bufferization.materialize_in_destination %15 in writable %reinterpret_cast_3 : (tensor<32x32xf16>, memref<32x32xf16, strided<[32, 1], offset: ?>>) -> () loc(#loc8)
    return loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc7 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":49:25)
#loc8 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":112:20)
#loc9 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":51:8)
#loc10 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":76:22)
#loc12 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":87:26)
#loc25 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":97:28)
#loc26 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":99:26)
#loc30 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":107:34)
#loc31 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":108:36)
#loc32 = loc("/path/to/loom/test/Triton/flashattn/flashattn.py":108:8)
#loc37 = loc(fused[#loc8, #loc9])

