#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0 + 1, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1 + 1)>
#map3 = affine_map<(d0) -> ((d0 ceildiv 8) ceildiv 8)>
#map4 = affine_map<(d0, d1, d2) -> (d0 * 1024 + d1 * 65536 + d2 * 8192)>
#map5 = affine_map<()[s0] -> (s0)>
#map6 = affine_map<(d0) -> (d0)>
#map7 = affine_map<(d0, d1) -> (d0, 0)>
#map8 = affine_map<()[s0] -> (s0 * 32)>
module {
  %0 = df.spatial_dim "x", 8
  %1 = df.spatial_dim "y", 8
  %2 = df.compute "cores", %0, %1 {map = #map}
  %3 = df.memory "L1", %0, %1 {map = #map}
  %4 = df.mux %2 : !df.compute, %3 : !df.memory, %0, %1 {map = #map}
  %5 = df.interconnects "horizontal_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = #map1} : !df.interconnect
  %6 = df.interconnects "vertical_links" %3 : !df.memory, %3 : !df.memory, %0, %1 {map = #map2} : !df.interconnect
  func.func @flashattn_fwd__d0i0_d1i0_f0(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to #map3(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %7 = tensor.empty() : tensor<32xf32>
          %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32xf32>) -> tensor<32xf32>
          %9 = tensor.empty() : tensor<32x32xf32>
          %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %11 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %12 = linalg.fill ins(%cst_2 : f32) outs(%7 : tensor<32xf32>) -> tensor<32xf32>
          %13 = affine.apply #map4(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [32, 32], strides: [32, 1] {tmd.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc = memref.alloc() : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %14 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16> to tensor<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %c0_3 = arith.constant 0 : index
          %c0_4 = arith.constant 0 : index
          %15:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0_3, %arg12 = %c0_4, %arg13 = %12, %arg14 = %8, %arg15 = %11) -> (index, index, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) {
            %21 = affine.apply #map5()[%arg12]
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%21], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_7 = memref.alloc() : memref<32x32xf16>
            memref.copy %reinterpret_cast_6, %alloc_7 : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %22 = bufferization.to_tensor %alloc_7 restrict writable : memref<32x32xf16> to tensor<32x32xf16>
            %23 = linalg.matmul ins(%14, %22 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %24 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%23, %10 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.mulf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %transposed = linalg.transpose ins(%24 : tensor<32x32xf32>) outs(%9 : tensor<32x32xf32>) permutation = [1, 0] 
            %reduced = linalg.reduce ins(%transposed : tensor<32x32xf32>) outs(%12 : tensor<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %44 = arith.maxnumf %in, %init : f32
                linalg.yield %44 : f32
              }
            %25 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%arg13, %reduced : tensor<32xf32>, tensor<32xf32>) outs(%arg13 : tensor<32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.maxnumf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %expanded_8 = tensor.expand_shape %25 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
            %26 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_8 : tensor<32x1xf32>) outs(%9 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            } -> tensor<32x32xf32>
            %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%24, %26 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%24 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.subf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = math.exp %in : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %transposed_9 = linalg.transpose ins(%28 : tensor<32x32xf32>) outs(%9 : tensor<32x32xf32>) permutation = [1, 0] 
            %29 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<32xf32>) -> tensor<32xf32>
            %reduced_10 = linalg.reduce ins(%transposed_9 : tensor<32x32xf32>) outs(%29 : tensor<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %44 = arith.addf %in, %init : f32
                linalg.yield %44 : f32
              }
            %30 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%arg13, %25 : tensor<32xf32>, tensor<32xf32>) outs(%arg13 : tensor<32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.subf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %31 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel"]} ins(%30 : tensor<32xf32>) outs(%30 : tensor<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = math.exp %in : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %32 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%arg14, %31 : tensor<32xf32>, tensor<32xf32>) outs(%arg14 : tensor<32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.mulf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %33 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%32, %reduced_10 : tensor<32xf32>, tensor<32xf32>) outs(%32 : tensor<32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.addf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %34 = affine.apply #map8()[%arg11]
            %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [32, 32], strides: [32, 1] {tmd.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_12 = memref.alloc() : memref<32x32xf16>
            memref.copy %reinterpret_cast_11, %alloc_12 : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %35 = bufferization.to_tensor %alloc_12 restrict writable : memref<32x32xf16> to tensor<32x32xf16>
            %36 = tensor.empty() : tensor<32x32xf16>
            %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%28 : tensor<32x32xf32>) outs(%36 : tensor<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %44 = arith.truncf %in : f32 to f16
              linalg.yield %44 : f16
            } -> tensor<32x32xf16>
            %expanded_13 = tensor.expand_shape %31 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
            %38 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_13 : tensor<32x1xf32>) outs(%9 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            } -> tensor<32x32xf32>
            %39 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg15, %38 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg15 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.mulf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %40 = linalg.matmul ins(%37, %35 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%39, %40 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%39 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.addf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %c32_14 = arith.constant 32 : index
            %42 = arith.addi %arg11, %c32_14 : index
            %c32_15 = arith.constant 32 : index
            %43 = arith.addi %arg12, %c32_15 : index
            scf.yield %42, %43, %25, %33, %41 : index, index, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>
          }
          %expanded = tensor.expand_shape %15#3 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
          %16 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<32x1xf32>) outs(%9 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x32xf32>
          %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%15#4, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%15#4 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %21 = arith.divf %in, %in_6 : f32
            linalg.yield %21 : f32
          } -> tensor<32x32xf32>
          %18 = tensor.empty() : tensor<32x32xf16>
          %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<32x32xf32>) outs(%18 : tensor<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %21 = arith.truncf %in : f32 to f16
            linalg.yield %21 : f16
          } -> tensor<32x32xf16>
          %20 = affine.apply #map4(%arg9, %arg7, %arg8)
          %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%20], sizes: [32, 32], strides: [32, 1] {tmd.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          bufferization.materialize_in_destination %19 in writable %reinterpret_cast_5 : (tensor<32x32xf16>, memref<32x32xf16, strided<[32, 1], offset: ?>>) -> ()
        } {tmd.mapped_to = "x"}
      } {tmd.mapped_to = "y"}
    }
    return
  }
  func.func @flashattn_fwd__d1i0_d0i0_f0(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf16> {tt.divisibility = 16 : i32}, %arg3: memref<*xf16> {tt.divisibility = 16 : i32}, %arg4: index, %arg5: index, %arg6: index) {
    affine.for %arg7 = 0 to #map3(%arg4) {
      affine.parallel (%arg8) = (0) to (8) {
        affine.parallel (%arg9) = (0) to (8) {
          %cst = arith.constant 1.000000e+00 : f32
          %cst_0 = arith.constant 0.176776692 : f32
          %cst_1 = arith.constant 0.000000e+00 : f32
          %cst_2 = arith.constant 0xFF800000 : f32
          %7 = tensor.empty() : tensor<32xf32>
          %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<32xf32>) -> tensor<32xf32>
          %9 = tensor.empty() : tensor<32x32xf32>
          %10 = linalg.fill ins(%cst_0 : f32) outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %11 = linalg.fill ins(%cst_1 : f32) outs(%9 : tensor<32x32xf32>) -> tensor<32x32xf32>
          %12 = linalg.fill ins(%cst_2 : f32) outs(%7 : tensor<32xf32>) -> tensor<32xf32>
          %13 = affine.apply #map4(%arg9, %arg7, %arg8)
          %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%13], sizes: [32, 32], strides: [32, 1] {tmd.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          %alloc = memref.alloc() : memref<32x32xf16>
          memref.copy %reinterpret_cast, %alloc : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
          %14 = bufferization.to_tensor %alloc restrict writable : memref<32x32xf16> to tensor<32x32xf16>
          %c0 = arith.constant 0 : index
          %c512 = arith.constant 512 : index
          %c32 = arith.constant 32 : index
          %c0_3 = arith.constant 0 : index
          %c0_4 = arith.constant 0 : index
          %15:5 = scf.for %arg10 = %c0 to %c512 step %c32 iter_args(%arg11 = %c0_3, %arg12 = %c0_4, %arg13 = %12, %arg14 = %8, %arg15 = %11) -> (index, index, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>) {
            %21 = affine.apply #map5()[%arg12]
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%21], sizes: [32, 32], strides: [512, 1] {tmd.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[512, 1], offset: ?>>
            %alloc_7 = memref.alloc() : memref<32x32xf16>
            memref.copy %reinterpret_cast_6, %alloc_7 : memref<32x32xf16, strided<[512, 1], offset: ?>> to memref<32x32xf16>
            %22 = bufferization.to_tensor %alloc_7 restrict writable : memref<32x32xf16> to tensor<32x32xf16>
            %23 = linalg.matmul ins(%14, %22 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %24 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%23, %10 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%23 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.mulf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %transposed = linalg.transpose ins(%24 : tensor<32x32xf32>) outs(%9 : tensor<32x32xf32>) permutation = [1, 0] 
            %reduced = linalg.reduce ins(%transposed : tensor<32x32xf32>) outs(%12 : tensor<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %44 = arith.maxnumf %in, %init : f32
                linalg.yield %44 : f32
              }
            %25 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%arg13, %reduced : tensor<32xf32>, tensor<32xf32>) outs(%arg13 : tensor<32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.maxnumf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %expanded_8 = tensor.expand_shape %25 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
            %26 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_8 : tensor<32x1xf32>) outs(%9 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            } -> tensor<32x32xf32>
            %27 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%24, %26 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%24 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.subf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %28 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%27 : tensor<32x32xf32>) outs(%27 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = math.exp %in : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %transposed_9 = linalg.transpose ins(%28 : tensor<32x32xf32>) outs(%9 : tensor<32x32xf32>) permutation = [1, 0] 
            %29 = linalg.fill ins(%cst_1 : f32) outs(%7 : tensor<32xf32>) -> tensor<32xf32>
            %reduced_10 = linalg.reduce ins(%transposed_9 : tensor<32x32xf32>) outs(%29 : tensor<32xf32>) dimensions = [0] 
              (%in: f32, %init: f32) {
                %44 = arith.addf %in, %init : f32
                linalg.yield %44 : f32
              }
            %30 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%arg13, %25 : tensor<32xf32>, tensor<32xf32>) outs(%arg13 : tensor<32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.subf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %31 = linalg.generic {indexing_maps = [#map6, #map6], iterator_types = ["parallel"]} ins(%30 : tensor<32xf32>) outs(%30 : tensor<32xf32>) {
            ^bb0(%in: f32, %out: f32):
              %44 = math.exp %in : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %32 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%arg14, %31 : tensor<32xf32>, tensor<32xf32>) outs(%arg14 : tensor<32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.mulf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %33 = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%32, %reduced_10 : tensor<32xf32>, tensor<32xf32>) outs(%32 : tensor<32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.addf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32xf32>
            %34 = affine.apply #map8()[%arg11]
            %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [32, 32], strides: [32, 1] {tmd.reuse = {sequential = [{depth = 3 : i64, iterator = "%arg10", reuse_type = "total_reuse", volume = 2048 : i64}], spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "total_reuse", volume = 2048 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "total_reuse", volume = 2048 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "total_reuse", volume = 2048 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
            %alloc_12 = memref.alloc() : memref<32x32xf16>
            memref.copy %reinterpret_cast_11, %alloc_12 : memref<32x32xf16, strided<[32, 1], offset: ?>> to memref<32x32xf16>
            %35 = bufferization.to_tensor %alloc_12 restrict writable : memref<32x32xf16> to tensor<32x32xf16>
            %36 = tensor.empty() : tensor<32x32xf16>
            %37 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%28 : tensor<32x32xf32>) outs(%36 : tensor<32x32xf16>) {
            ^bb0(%in: f32, %out: f16):
              %44 = arith.truncf %in : f32 to f16
              linalg.yield %44 : f16
            } -> tensor<32x32xf16>
            %expanded_13 = tensor.expand_shape %31 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
            %38 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded_13 : tensor<32x1xf32>) outs(%9 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
            ^bb0(%in: f32, %out: f32):
              linalg.yield %in : f32
            } -> tensor<32x32xf32>
            %39 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg15, %38 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg15 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.mulf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %40 = linalg.matmul ins(%37, %35 : tensor<32x32xf16>, tensor<32x32xf16>) outs(%11 : tensor<32x32xf32>) -> tensor<32x32xf32>
            %41 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%39, %40 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%39 : tensor<32x32xf32>) {
            ^bb0(%in: f32, %in_16: f32, %out: f32):
              %44 = arith.addf %in, %in_16 : f32
              linalg.yield %44 : f32
            } -> tensor<32x32xf32>
            %c32_14 = arith.constant 32 : index
            %42 = arith.addi %arg11, %c32_14 : index
            %c32_15 = arith.constant 32 : index
            %43 = arith.addi %arg12, %c32_15 : index
            scf.yield %42, %43, %25, %33, %41 : index, index, tensor<32xf32>, tensor<32xf32>, tensor<32x32xf32>
          }
          %expanded = tensor.expand_shape %15#3 [[0, 1]] output_shape [32, 1] : tensor<32xf32> into tensor<32x1xf32>
          %16 = linalg.generic {indexing_maps = [#map7, #map], iterator_types = ["parallel", "parallel"]} ins(%expanded : tensor<32x1xf32>) outs(%9 : tensor<32x32xf32>) attrs =  {broadcastDims = array<i64: 1>} {
          ^bb0(%in: f32, %out: f32):
            linalg.yield %in : f32
          } -> tensor<32x32xf32>
          %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%15#4, %16 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%15#4 : tensor<32x32xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %21 = arith.divf %in, %in_6 : f32
            linalg.yield %21 : f32
          } -> tensor<32x32xf32>
          %18 = tensor.empty() : tensor<32x32xf16>
          %19 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%17 : tensor<32x32xf32>) outs(%18 : tensor<32x32xf16>) {
          ^bb0(%in: f32, %out: f16):
            %21 = arith.truncf %in : f32 to f16
            linalg.yield %21 : f16
          } -> tensor<32x32xf16>
          %20 = affine.apply #map4(%arg9, %arg7, %arg8)
          %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%20], sizes: [32, 32], strides: [32, 1] {tmd.reuse = {spatial = [{depth = 1 : i64, iterator = "%arg8", mapped_to = "x", reuse_type = "no_reuse", volume = 0 : i64}, {depth = 2 : i64, iterator = "%arg9", mapped_to = "y", reuse_type = "no_reuse", volume = 0 : i64}], temporal = [{depth = 0 : i64, iterator = "%arg7", reuse_type = "no_reuse", volume = 0 : i64}]}} : memref<*xf16> to memref<32x32xf16, strided<[32, 1], offset: ?>>
          bufferization.materialize_in_destination %19 in writable %reinterpret_cast_5 : (tensor<32x32xf16>, memref<32x32xf16, strided<[32, 1], offset: ?>>) -> ()
        } {tmd.mapped_to = "y"}
      } {tmd.mapped_to = "x"}
    }
    return
  }
}

