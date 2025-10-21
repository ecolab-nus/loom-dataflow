module {
  func.func @flashattn_fwd(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.parallel (%arg7) = (0) to (%arg4) {
      %cst = arith.constant 1.000000e+00 : f32
      %cst_0 = arith.constant 0.176776692 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 0xFF800000 : f32
      %0 = tensor.empty() : tensor<32xf32>
      %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
      %2 = tensor.empty() : tensor<32x32xf32>
      %3 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %4 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<32x32xf32>) -> tensor<32x32xf32>
      %5 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
      %6 = affine.apply affine_map<(d0) -> (d0 * 1024)>(%arg7)
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%6], sizes: [32, 32], strides: [32, 1] : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
      %alloc = memref.alloc() : memref<32x32xf16>
      memref.copy %reinterpret_cast, %alloc : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
      %7 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16> to tensor<32x32xf16>
      %c0 = arith.constant 0 : index
      %c512 = arith.constant 512 : index
      %c32 = arith.constant 32 : index
      %c0_3 = arith.constant 0 : index
      %c0_4 = arith.constant 0 : index
      %8:5 = scf.for %arg8 = %c0 to %c512 step %c32 iter_args(%arg9 = %c0_3, %arg10 = %c0_4, %arg11 = %5, %arg12 = %1, %arg13 = %4) -> (index, index, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) {
        %14 = affine.apply affine_map<()[s0] -> (s0)>()[%arg10]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%14], sizes: [32, 32], strides: [512, 1] : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<32x32xf16>
        memref.copy %reinterpret_cast_6, %alloc_7 : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
        %15 = bufferization.to_tensor %alloc_7 restrict writable : memref<32x32xf16> to tensor<32x32xf16>
        %16 = linalg.matmul ins(%7, %15 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%16, %3 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%16 : tensor<32x32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %37 = arith.mulf %in, %in_16 : f32
          linalg.yield %37 : f32
        } -> tensor<32x32xf32>
        %transposed = linalg.transpose ins(%17 : tensor<32x32xf32>) outs(%2 : tensor<32x32xf32>) permutation = [1, 0] 
        %reduced = linalg.reduce ins(%transposed : tensor<32x32xf32>) outs(%5 : tensor<32xf32>) dimensions = [0] 
          (%in: f32, %init: f32) {
            %37 = arith.maxnumf %in, %init : f32
            linalg.yield %37 : f32
          }
        %18 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg11, %reduced : tensor<32xf32>, tensor<32xf32>) outs(%arg11 : tensor<32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %37 = arith.maxnumf %in, %in_16 : f32
          linalg.yield %37 : f32
        } -> tensor<32xf32>
        %expanded_8 = tensor.expand_shape %18 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded_8 : tensor<32x1xf32>) outs(%2 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<32x32xf32>
        %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%17, %19 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%17 : tensor<32x32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %37 = arith.subf %in, %in_16 : f32
          linalg.yield %37 : f32
        } -> tensor<32x32xf32>
        %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<32x32xf32>) outs(%20 : tensor<32x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          %37 = math.exp %in : f32
          linalg.yield %37 : f32
        } -> tensor<32x32xf32>
        %transposed_9 = linalg.transpose ins(%21 : tensor<32x32xf32>) outs(%2 : tensor<32x32xf32>) permutation = [1, 0] 
        %22 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
        %reduced_10 = linalg.reduce ins(%transposed_9 : tensor<32x32xf32>) outs(%22 : tensor<32xf32>) dimensions = [0] 
          (%in: f32, %init: f32) {
            %37 = arith.addf %in, %init : f32
            linalg.yield %37 : f32
          }
        %23 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg11, %18 : tensor<32xf32>, tensor<32xf32>) outs(%arg11 : tensor<32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %37 = arith.subf %in, %in_16 : f32
          linalg.yield %37 : f32
        } -> tensor<32xf32>
        %24 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%23 : tensor<32xf32>) outs(%23 : tensor<32xf32>) {
        ^bb0(%in: f32, %out: f32):
          %37 = math.exp %in : f32
          linalg.yield %37 : f32
        } -> tensor<32xf32>
        %25 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg12, %24 : tensor<32xf32>, tensor<32xf32>) outs(%arg12 : tensor<32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %37 = arith.mulf %in, %in_16 : f32
          linalg.yield %37 : f32
        } -> tensor<32xf32>
        %26 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%25, %reduced_10 : tensor<32xf32>, tensor<32xf32>) outs(%25 : tensor<32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %37 = arith.addf %in, %in_16 : f32
          linalg.yield %37 : f32
        } -> tensor<32xf32>
        %27 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg9]
        %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%27], sizes: [32, 32], strides: [32, 1] : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
        %alloc_12 = memref.alloc() : memref<32x32xf16>
        memref.copy %reinterpret_cast_11, %alloc_12 : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
        %28 = bufferization.to_tensor %alloc_12 restrict writable : memref<32x32xf16> to tensor<32x32xf16>
        %29 = tensor.empty() : tensor<32x32xf16>
        %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21 : tensor<32x32xf32>) outs(%29 : tensor<32x32xf16>) {
        ^bb0(%in: f32, %out: f16):
          %37 = arith.truncf %in : f32 to f16
          linalg.yield %37 : f16
        } -> tensor<32x32xf16>
        %expanded_13 = tensor.expand_shape %24 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
        %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded_13 : tensor<32x1xf32>) outs(%2 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        } -> tensor<32x32xf32>
        %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg13, %31 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg13 : tensor<32x32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %37 = arith.mulf %in, %in_16 : f32
          linalg.yield %37 : f32
        } -> tensor<32x32xf32>
        %33 = linalg.matmul ins(%30, %28 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>
        %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32, %33 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%32 : tensor<32x32xf32>) {
        ^bb0(%in: f32, %in_16: f32, %out: f32):
          %37 = arith.addf %in, %in_16 : f32
          linalg.yield %37 : f32
        } -> tensor<32x32xf32>
        %c32_14 = arith.constant 32 : index
        %35 = arith.addi %arg9, %c32_14 : index
        %c32_15 = arith.constant 32 : index
        %36 = arith.addi %arg10, %c32_15 : index
        scf.yield %35, %36, %18, %26, %34 : index, index, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>
      }
      %expanded = tensor.expand_shape %8#3 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
      %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<32x1xf32>) outs(%2 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<32x32xf32>
      %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8#4, %9 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%8#4 : tensor<32x32xf32>) {
      ^bb0(%in: f32, %in_6: f32, %out: f32):
        %14 = arith.divf %in, %in_6 : f32
        linalg.yield %14 : f32
      } -> tensor<32x32xf32>
      %11 = tensor.empty() : tensor<32x32xf16>
      %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<32x32xf32>) outs(%11 : tensor<32x32xf16>) {
      ^bb0(%in: f32, %out: f16):
        %14 = arith.truncf %in : f32 to f16
        linalg.yield %14 : f16
      } -> tensor<32x32xf16>
      %13 = affine.apply affine_map<(d0) -> (d0 * 1024)>(%arg7)
      %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%13], sizes: [32, 32], strides: [32, 1] : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
      bufferization.materialize_in_destination %12 in writable %reinterpret_cast_5 : (tensor<32x32xf16>, memref<32x32xf16, strided<[32, 1], offset: ?>>) -> ()
    }
    return
  }
}
