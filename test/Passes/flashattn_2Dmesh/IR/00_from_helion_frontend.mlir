// affine_maps
#map = affine_map<()[s0] -> (4096 ceildiv s0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>

module {
  module {
    func.func @attention(%arg0: memref<1x128x4096xf16>, %arg1: memref<1x4096x128xf16>, %arg2: memref<1x4096x128xf16>, %arg3: memref<1x4096x128xf16>) {
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.12751743 : f16 // qk_scale
      %c0_i64 = arith.constant 0 : i64
      %cst_1 = arith.constant 0.000000e+00 : f16
      %cst_2 = arith.constant 1.000000e+00 : f16
      %cst_3 = arith.constant 0xFC00 : f16 // -inf
      // Manually added constraint space access
      %0 = loom.sym @block_size_0 : index // BB
      %1 = loom.sym @block_size_1 : index // BM
      %2 = loom.sym @block_size_2 : index // BN
      affine.parallel (%arg4, %arg5) = (0, 0) to (1 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = tensor.empty(%0, %1) : tensor<?x?xf16>
        %4 = linalg.fill ins(%cst_3 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16> // m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float16)
        %5 = linalg.fill ins(%cst_2 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16> // l_i = torch.full_like(m_i, 1.0)
        %6 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
        %7 = linalg.fill ins(%cst_1 : f16) outs(%6 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16> // acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float16)
        %8 = arith.muli %arg4, %0 : index // iter_B * BB
        %9 = arith.muli %arg5, %1 : index // iter_M * BM
        %subview = memref.subview %arg2[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<1x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> // subview of Q block with size [BB, BM, dim]
        %10 = bufferization.to_tensor %subview : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16> // buffer Q to tensor
        %11:3 = affine.for %arg6 = 0 to #map()[%2] iter_args(%arg7 = %4, %arg8 = %5, %arg9 = %7) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) { // iter on N dimension
          %14 = arith.muli %arg6, %2 : index // iter_N * BN
          // NOTE: %arg0 is K^T
          %subview_5 = memref.subview %arg0[%8, 0, %14] [%0, 128, %2] [1, 1, 1] : memref<1x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> // subview of K^T block with size [BB, dim, BN]
          %15 = bufferization.to_tensor %subview_5 : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf16> // buffer K^T to tensor
          %16 = arith.index_cast %0 : index to i64
          %17 = arith.cmpi eq, %16, %16 : i64
          cf.assert %17, "mismatching contracting dimension"
          %18 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
          %19 = linalg.fill ins(%cst_1 : f16) outs(%18 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
          %20 = linalg.batch_matmul ins(%10, %15 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%19 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16> // Q @ K^T
          //** torch.amax(qk, -1) **/
          %21 = tensor.empty(%0, %1) : tensor<?x?xi64>
          %22 = linalg.fill ins(%c0_i64 : i64) outs(%21 : tensor<?x?xi64>) -> tensor<?x?xi64>
          // %20: qk, %4: -inf, %22: index of max
          %23 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%20 : tensor<?x?x?xf16>) outs(%4 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %43 = arith.maximumf %in, %out : f16
            linalg.yield %43 : f16
          } -> (tensor<?x?xf16>)
          //** torch.amax(qk, -1) * qk_scale **/
          %24 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%23 : tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %42 = arith.mulf %in, %cst_0 : f16
            linalg.yield %42 : f16
          } -> tensor<?x?xf16>
          //** m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale) **/
          %25 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg7, %24 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_11: f16, %out: f16):
            %41 = arith.cmpf ogt, %in, %in_11 : f16
            %42 = arith.select %41, %in, %in_11 : f16
            linalg.yield %42 : f16
          } -> tensor<?x?xf16>
          // qk *= qk_scale
          %26 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<?x?x?xf16>) outs(%18 : tensor<?x?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %42 = arith.mulf %in, %cst_0 : f16
            linalg.yield %42 : f16
          } -> tensor<?x?x?xf16>
          // m_ij[:, :, None]
          %extracted_slice_6 = tensor.extract_slice %25[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
          %expanded_7 = tensor.expand_shape %extracted_slice_6 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
          // qk -= m_ij[:, :, None]
          %27 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%26, %expanded_7 : tensor<?x?x?xf16>, tensor<?x?x1xf16>) outs(%18 : tensor<?x?x?xf16>) {
          ^bb0(%in: f16, %in_11: f16, %out: f16):
            %41 = arith.subf %in, %in_11 : f16
            linalg.yield %41 : f16
          } -> tensor<?x?x?xf16>
          // p = torch.exp2(qk)
          %28 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27 : tensor<?x?x?xf16>) outs(%18 : tensor<?x?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %41 = math.powf %cst, %in : f16
            linalg.yield %41 : f16
          } -> tensor<?x?x?xf16>
          // l_ij = torch.sum(p, -1)
          %29 = linalg.fill ins(%cst_1 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %30 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28 : tensor<?x?x?xf16>) outs(%29 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %41 = arith.addf %in, %out : f16
            linalg.yield %41 : f16
          } -> tensor<?x?xf16>
          // intermediate = m_i -m_ij
          %31 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg7, %25 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_11: f16, %out: f16):
            %41 = arith.subf %in, %in_11 : f16
            linalg.yield %41 : f16
          } -> tensor<?x?xf16>
          // alpha = torch.exp2(intermediate)
          %32 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%31 : tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %41 = math.powf %cst, %in : f16
            linalg.yield %41 : f16
          } -> tensor<?x?xf16>
          // l_i *= alpha
          %33 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%arg8, %32 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_11: f16, %out: f16):
            %41 = arith.mulf %in, %in_11 : f16
            linalg.yield %41 : f16
          } -> tensor<?x?xf16>
          // l_i += l_ij
          %34 = linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%33, %30 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_11: f16, %out: f16):
            %41 = arith.addf %in, %in_11 : f16
            linalg.yield %41 : f16
          } -> tensor<?x?xf16>
          // alpha[:, :, None]
          %extracted_slice_8 = tensor.extract_slice %32[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
          %expanded_9 = tensor.expand_shape %extracted_slice_8 [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
          // acc *= alpha[:, :, None]
          %35 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg9, %expanded_9 : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%6 : tensor<?x?x128xf16>) {
          ^bb0(%in: f16, %in_11: f16, %out: f16):
            %41 = arith.mulf %in, %in_11 : f16
            linalg.yield %41 : f16
          } -> tensor<?x?x128xf16>
          %subview_10 = memref.subview %arg1[%8, %14, 0] [%0, %2, 128] [1, 1, 1] : memref<1x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> // subview of V block with size [BB, BN, dim]
          %36 = bufferization.to_tensor %subview_10 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16> // buffer V to tensor
          cf.assert %17, "mismatching contracting dimension"
          %37 = arith.index_cast %2 : index to i64
          %38 = arith.cmpi eq, %37, %37 : i64
          cf.assert %38, "mismatching contracting dimension"
          // p @ v
          %39 = linalg.batch_matmul ins(%28, %36 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%7 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
          // acc += p @ v
          %40 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %35 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%6 : tensor<?x?x128xf16>) {
          ^bb0(%in: f16, %in_11: f16, %out: f16):
            %41 = arith.addf %in, %in_11 : f16
            linalg.yield %41 : f16
          } -> tensor<?x?x128xf16>
          // yield m_ij as m_i, l_i, acc
          affine.yield %25, %34, %40 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
        }
        // l_i[:, :, None]
        %extracted_slice = tensor.extract_slice %11#1[0, 0] [%0, %1] [1, 1] : tensor<?x?xf16> to tensor<?x?xf16>
        %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2]] output_shape [%0, %1, 1] : tensor<?x?xf16> into tensor<?x?x1xf16>
        // acc /= l_i[:, :, None]
        %12 = linalg.generic {indexing_maps = [#map1, #map4, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11#2, %expanded : tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%6 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %14 = arith.divf %in, %in_5 : f16
          linalg.yield %14 : f16
        } -> tensor<?x?x128xf16>
        %subview_4 = memref.subview %arg3[%8, %9, 0] [%0, %1, 128] [1, 1, 1] : memref<1x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> // subview of O block with size [BB, BM, dim]
        %13 = bufferization.to_buffer %12 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        memref.copy %13, %subview_4 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      }
      return
    }
  }
}