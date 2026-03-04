// affine_maps
#map = affine_map<()[s0] -> (4096 ceildiv s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>

module {
  module {
    // Manually added constraint space
    loom.constraint_space @constraints {
      %bb = loom.symbolic_var "BB" : index
      %bm = loom.symbolic_var "BM" : index
      %bn = loom.symbolic_var "BN" : index

      loom.range %bb [0, 32]
      loom.range %bm [0, 4096]
      loom.range %bn [0, 4096]
    }
    func.func @attention(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64 // qk_scale
      %c0_i64 = arith.constant 0 : i64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32 // -inf
      // Manually added constraint space access
      %0 = loom.get_symbolic_block_size @constraints::@BB : index
      %1 = loom.get_symbolic_block_size @constraints::@BM : index
      %2 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4, %arg5) = (0, 0) to (32 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
        %4 = linalg.fill ins(%cst_3 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32> // m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        %5 = linalg.fill ins(%cst_2 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32> // l_i = torch.full_like(m_i, 1.0)
        %6 = tensor.empty(%0, %1) : tensor<?x?x128xf32>
        %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32> // acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        %8 = arith.muli %arg4, %0 : index // iter_B * BB
        %9 = arith.muli %arg5, %1 : index // iter_M * BM
        %subview = memref.subview %arg2[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>> // subview of Q block with size [BB, BM, dim]
        %10 = bufferization.to_tensor %subview : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf32> // buffer Q to tensor
        %11:3 = affine.for %arg6 = 0 to #map()[%2] iter_args(%arg7 = %4, %arg8 = %5, %arg9 = %7) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) { // iter on N dimension
          %14 = arith.muli %arg6, %2 : index // iter_N * BN
          // NOTE: %arg0 is K^T
          %subview_5 = memref.subview %arg0[%8, 0, %14] [%0, 128, %2] [1, 1, 1] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>> // subview of K^T block with size [BB, dim, BN]
          %15 = bufferization.to_tensor %subview_5 : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf32> // buffer K^T to tensor
          %16 = arith.index_cast %0 : index to i64
          %17 = arith.cmpi eq, %16, %16 : i64
          cf.assert %17, "mismatching contracting dimension"
          %18 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
          %19 = linalg.fill ins(%cst_1 : f32) outs(%18 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %20 = linalg.batch_matmul ins(%10, %15 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%19 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> // Q @ K^T
          //** torch.amax(qk, -1) **/
          %21 = tensor.empty(%0, %1) : tensor<?x?xi64>
          %22 = linalg.fill ins(%c0_i64 : i64) outs(%21 : tensor<?x?xi64>) -> tensor<?x?xi64>
          // %20: qk, %4: -inf, %22: index of max
          %23:2 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%20 : tensor<?x?x?xf32>) outs(%4, %22 : tensor<?x?xf32>, tensor<?x?xi64>) {
          ^bb0(%in: f32, %out: f32, %out_11: i64):
            %41 = linalg.index 2 : index
            %42 = arith.index_cast %41 : index to i64
            %43 = arith.maximumf %in, %out : f32
            %44 = arith.cmpf ogt, %in, %out : f32
            %45 = arith.select %44, %42, %out_11 : i64
            linalg.yield %43, %45 : f32, i64
          } -> (tensor<?x?xf32>, tensor<?x?xi64>)
          //** torch.amax(qk, -1) * qk_scale **/
          %24 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%23#0 : tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %41 = arith.truncf %cst_0 : f64 to f32
            %42 = arith.mulf %in, %41 : f32
            linalg.yield %42 : f32
          } -> tensor<?x?xf32>
          //** m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale) **/
          %25 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg7, %24 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_11: f32, %out: f32):
            %41 = arith.cmpf ogt, %in, %in_11 : f32
            %42 = arith.select %41, %in, %in_11 : f32
            linalg.yield %42 : f32
          } -> tensor<?x?xf32>
          // qk *= qk_scale
          %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<?x?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %41 = arith.truncf %cst_0 : f64 to f32
            %42 = arith.mulf %in, %41 : f32
            linalg.yield %42 : f32
          } -> tensor<?x?x?xf32>
          // m_ij[:, :, None]
          %extracted_slice_6 = tensor.extract_slice %25[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %expanded_7 = tensor.expand_shape %extracted_slice_6 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf32> into tensor<?x?x1xf32>
          // qk -= m_ij[:, :, None]
          %27 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %expanded_7 : tensor<?x?x?xf32>, tensor<?x?x1xf32>) outs(%18 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_11: f32, %out: f32):
            %41 = arith.subf %in, %in_11 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?x?xf32>
          // p = torch.exp2(qk)
          %28 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27 : tensor<?x?x?xf32>) outs(%18 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %41 = math.powf %cst, %in : f32
            linalg.yield %41 : f32
          } -> tensor<?x?x?xf32>
          // l_ij = torch.sum(p, -1)
          %29 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %30 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28 : tensor<?x?x?xf32>) outs(%29 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %41 = arith.addf %in, %out : f32
            linalg.yield %41 : f32
          } -> tensor<?x?xf32>
          // intermediate = m_i -m_ij
          %31 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg7, %25 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_11: f32, %out: f32):
            %41 = arith.subf %in, %in_11 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?xf32>
          // alpha = torch.exp2(intermediate)
          %32 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%31 : tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %41 = math.powf %cst, %in : f32
            linalg.yield %41 : f32
          } -> tensor<?x?xf32>
          // l_i *= alpha
          %33 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg8, %32 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_11: f32, %out: f32):
            %41 = arith.mulf %in, %in_11 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?xf32>
          // l_i += l_ij
          %34 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%33, %30 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_11: f32, %out: f32):
            %41 = arith.addf %in, %in_11 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?xf32>
          // alpha[:, :, None]
          %extracted_slice_8 = tensor.extract_slice %32[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %expanded_9 = tensor.expand_shape %extracted_slice_8 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf32> into tensor<?x?x1xf32>
          // acc *= alpha[:, :, None]
          %35 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %expanded_9 : tensor<?x?x128xf32>, tensor<?x?x1xf32>) outs(%6 : tensor<?x?x128xf32>) {
          ^bb0(%in: f32, %in_11: f32, %out: f32):
            %41 = arith.mulf %in, %in_11 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?x128xf32>
          %subview_10 = memref.subview %arg1[%8, %14, 0] [%0, %2, 128] [1, 1, 1] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>> // subview of V block with size [BB, BN, dim]
          %36 = bufferization.to_tensor %subview_10 : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf32> // buffer V to tensor
          cf.assert %17, "mismatching contracting dimension"
          %37 = arith.index_cast %2 : index to i64
          %38 = arith.cmpi eq, %37, %37 : i64
          cf.assert %38, "mismatching contracting dimension"
          // p @ v
          %39 = linalg.batch_matmul ins(%28, %36 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%7 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
          // acc += p @ v
          %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %35 : tensor<?x?x128xf32>, tensor<?x?x128xf32>) outs(%6 : tensor<?x?x128xf32>) {
          ^bb0(%in: f32, %in_11: f32, %out: f32):
            %41 = arith.addf %in, %in_11 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?x128xf32>
          // yield m_ij as m_i, l_i, acc
          affine.yield %25, %34, %40 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
        }
        // l_i[:, :, None]
        %extracted_slice = tensor.extract_slice %11#1[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf32> into tensor<?x?x1xf32>
        // acc /= l_i[:, :, None]
        %12 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11#2, %expanded : tensor<?x?x128xf32>, tensor<?x?x1xf32>) outs(%6 : tensor<?x?x128xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %14 = arith.divf %in, %in_5 : f32
          linalg.yield %14 : f32
        } -> tensor<?x?x128xf32>
        %subview_4 = memref.subview %arg3[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>> // subview of O block with size [BB, BM, dim]
        %13 = bufferization.to_buffer %12 : tensor<?x?x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
        memref.copy %13, %subview_4 : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
      }
      return
    }
  }
}
