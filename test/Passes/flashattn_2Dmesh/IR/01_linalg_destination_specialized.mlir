module {
  module {
    loom.constraint_space @constraints {
      %0 = loom.symbolic_var "BB" : index
      %1 = loom.symbolic_var "BM" : index
      %2 = loom.symbolic_var "BN" : index
      loom.range %0[0, 2]
      loom.range %1[0, 4096]
      loom.range %2[0, 4096]
    }
    func.func @attention(%arg0: memref<2x4096x4096xf32>, %arg1: memref<2x4096x4096xf32>, %arg2: memref<2x4096x4096xf32>, %arg3: memref<2x4096x4096xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.022542110000000001 : f64 // qk_scale
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32 // -inf
      %0 = loom.get_symbolic_block_size @constraints::@BB : index
      %1 = loom.get_symbolic_block_size @constraints::@BM : index
      %2 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4, %arg5) = (0, 0) to (2 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
        %4 = linalg.fill ins(%cst_3 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32> // m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        %5 = linalg.fill ins(%cst_2 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32> // l_i = torch.full_like(m_i, 1.0)
        %6 = tensor.empty(%0, %1) : tensor<?x?x4096xf32>
        %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32> // acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        %8 = arith.muli %arg4, %0 : index // iter_B * BB
        %9 = arith.muli %arg5, %1 : index // iter_M * BM
        %subview = memref.subview %arg2[%8, %9, 0] [%0, %1, 4096] [1, 1, 1] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %10 = bufferization.to_tensor %subview : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>> to tensor<?x?x4096xf32>
        %11:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %4, %arg8 = %5, %arg9 = %7) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>) {
          %14 = arith.muli %arg6, %2 : index // iter_N * BN
          %subview_5 = memref.subview %arg0[%8, 0, %14] [%0, 4096, %2] [1, 1, 1] : memref<2x4096x4096xf32> to memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>>
          %15 = bufferization.to_tensor %subview_5 : memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>> to tensor<?x4096x?xf32>
          %16 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
          %17 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          // Q @ K^T
          %18 = linalg.batch_matmul ins(%10, %15 : tensor<?x?x4096xf32>, tensor<?x4096x?xf32>) outs(%17 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          // torch.amax(qk, -1)
          %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%18 : tensor<?x?x?xf32>) outs(%4 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %29 = arith.maximumf %in, %out : f32
            linalg.yield %29 : f32
          } -> tensor<?x?xf32>
          // m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
          %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %19 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_9: f32, %out: f32):
            %29 = arith.truncf %cst_0 : f64 to f32
            %30 = arith.mulf %in_9, %29 : f32
            %31 = arith.cmpf ogt, %in, %30 : f32
            %32 = arith.select %31, %in, %30 : f32
            linalg.yield %32 : f32
          } -> tensor<?x?xf32>
          %extracted_slice_6 = tensor.extract_slice %20[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          // p = torch.exp2(qk)
          %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %extracted_slice_6 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_9: f32, %out: f32):
            %29 = arith.truncf %cst_0 : f64 to f32
            %30 = arith.mulf %in, %29 : f32
            %31 = arith.subf %30, %in_9 : f32
            %32 = math.powf %cst, %31 : f32
            linalg.yield %32 : f32
          } -> tensor<?x?x?xf32>
          %22 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
          // l_ij = torch.sum(p, -1)
          %23 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%21 : tensor<?x?x?xf32>) outs(%22 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %29 = arith.addf %in, %out : f32
            linalg.yield %29 : f32
          } -> tensor<?x?xf32>
          // alpha = 2 ^ (m_i - m_ij)
          %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %20 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_9: f32, %out: f32):
            %29 = arith.subf %in, %in_9 : f32
            %30 = math.powf %cst, %29 : f32
            linalg.yield %30 : f32
          } -> tensor<?x?xf32>
          // l_i = alpha * l_i + l_ij, in-place substitute l_i should be observed by OSB
          %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %24, %23 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
            %29 = arith.mulf %in, %in_9 : f32
            %30 = arith.addf %29, %in_10 : f32
            linalg.yield %30 : f32
          } -> tensor<?x?xf32>
          // alpha[:, :]
          %extracted_slice_7 = tensor.extract_slice %24[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %subview_8 = memref.subview %arg1[%8, %14, 0] [%0, %2, 4096] [1, 1, 1] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          %26 = bufferization.to_tensor %subview_8 : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>> to tensor<?x?x4096xf32>
          // p @ v
          %27 = linalg.batch_matmul ins(%21, %26 : tensor<?x?x?xf32>, tensor<?x?x4096xf32>) outs(%7 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
          // acc = acc * alpha + p @ v
          %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%27, %arg9, %extracted_slice_7 : tensor<?x?x4096xf32>, tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?x4096xf32>) {
          ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
            %29 = arith.mulf %in_9, %in_10 : f32
            %30 = arith.addf %in, %29 : f32
            linalg.yield %30 : f32
          } -> tensor<?x?x4096xf32>
          affine.yield %20, %25, %28 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>
        }
        // l_i
        %extracted_slice = tensor.extract_slice %11#1[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        // acc /= l_i
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11#2, %extracted_slice : tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?x4096xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %14 = arith.divf %in, %in_5 : f32
          linalg.yield %14 : f32
        } -> tensor<?x?x4096xf32>
        %subview_4 = memref.subview %arg3[%8, %9, 0] [%0, %1, 4096] [1, 1, 1] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %13 = bufferization.to_buffer %12 : tensor<?x?x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        memref.copy %13, %subview_4 : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
      }
      return
    }
  }
}
