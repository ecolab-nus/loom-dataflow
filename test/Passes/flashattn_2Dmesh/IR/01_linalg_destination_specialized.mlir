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
      %cst_0 = arith.constant 0.022542110000000001 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %0 = loom.get_symbolic_block_size @constraints::@BB : index
      %1 = loom.get_symbolic_block_size @constraints::@BM : index
      %2 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4, %arg5) = (0, 0) to (2 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %4 = loom.init_tensor %3[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %5 = linalg.fill ins(%cst_3 : f32) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32> // m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        %6 = linalg.fill ins(%cst_2 : f32) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32> // l_i = torch.full_like(m_i, 1.0)
        %7 = loom.alloc [%0, %1, 4096] on @L1 : memref<?x?x4096xf32>
        %8 = loom.init_tensor %7[%0, %1, 4096] : memref<?x?x4096xf32> -> tensor<?x?x4096xf32> // out or acc = torch.empty_like(q_view)
        %9 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32> // set out or acc to 0
        %10 = arith.muli %arg4, %0 : index
        %11 = arith.muli %arg5, %1 : index
        %12 = loom.subview %arg2[%10, %11, 0] [%0, %1, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %13 = loom.alloc [%0, %1, 4096] on @L1 : memref<?x?x4096xf32>
        %14 = loom.copy_to_tensor %12, %13, interconnect : [], broadcast : [1, 1] : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
        %15:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %5, %arg8 = %6, %arg9 = %9) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>) {
          %18 = arith.muli %arg6, %2 : index
          %19 = loom.subview %arg0[%10, 0, %18] [%0, 4096, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>>
          %20 = loom.alloc [%0, 4096, %2] on @L1 : memref<?x4096x?xf32>
          %21 = loom.copy_to_tensor %19, %20, interconnect : [], broadcast : [1, 1] : memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x4096x?xf32> -> tensor<?x4096x?xf32>
          %22 = loom.alloc [%0, %1, %2] on @L1 : memref<?x?x?xf32>
          %23 = loom.init_tensor %22[%0, %1, %2] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
          %24 = linalg.fill ins(%cst_1 : f32) outs(%23 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %25 = linalg.batch_matmul ins(%14, %21 : tensor<?x?x4096xf32>, tensor<?x4096x?xf32>) outs(%24 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          // %26 = torch.max(qk, -1)
          %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%25 : tensor<?x?x?xf32>) outs(%5 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %38 = arith.maximumf %in, %out : f32
            linalg.yield %38 : f32
          } -> tensor<?x?xf32>
          // %27: m_ij = torch.maximum(m_i(%arg7), torch.amax(qk, -1) * qk_scale)
          %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %26 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %38 = arith.truncf %cst_0 : f64 to f32
            %39 = arith.mulf %in_6, %38 : f32 // torch.amax(qk, -1) * qk_scale
            %40 = arith.cmpf ogt, %in, %39 : f32
            %41 = arith.select %40, %in, %39 : f32
            linalg.yield %41 : f32
          } -> tensor<?x?xf32>
          // m_ij[:, :] ?
          %extracted_slice_4 = tensor.extract_slice %27[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          // %28 = p, in-place substitute qk
          %28 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25, %extracted_slice_4 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%23 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %38 = arith.truncf %cst_0 : f64 to f32
            %39 = arith.mulf %in, %38 : f32 // qk * qk_scale
            %40 = arith.subf %39, %in_6 : f32 // qk * qk_scale - m_ij
            %41 = math.powf %cst, %40 : f32 // 2 ^ (qk * qk_scale - m_ij)
            linalg.yield %41 : f32
          } -> tensor<?x?x?xf32>
          // l_ij = torch.sum(p, -1)
          %29 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%28 : tensor<?x?x?xf32>) outs(%29 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %38 = arith.addf %in, %out : f32
            linalg.yield %38 : f32
          } -> tensor<?x?xf32>
          // alpha = 2 ^ (m_i - m_ij)
          %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %27 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_6: f32, %out: f32):
            %38 = arith.subf %in, %in_6 : f32
            %39 = math.powf %cst, %38 : f32
            linalg.yield %39 : f32
          } -> tensor<?x?xf32>
          // l_i = alpha * l_i + l_ij, in-place substitute l_i should be observed by OSB
          %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %31, %30 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_6: f32, %in_7: f32, %out: f32):
            %38 = arith.mulf %in, %in_6 : f32
            %39 = arith.addf %38, %in_7 : f32
            linalg.yield %39 : f32
          } -> tensor<?x?xf32>
          // alpha[:, :]
          %extracted_slice_5 = tensor.extract_slice %31[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
          %33 = loom.subview %arg1[%10, %18, 0] [%0, %2, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          %34 = loom.alloc [%0, %2, 4096] on @L1 : memref<?x?x4096xf32>
          %35 = loom.copy_to_tensor %33, %34, interconnect : [], broadcast : [1, 1] : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
          // p @ v
          %36 = linalg.batch_matmul ins(%28, %35 : tensor<?x?x?xf32>, tensor<?x?x4096xf32>) outs(%9 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
          // acc = acc * alpha + p @ v
          %37 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %arg9, %extracted_slice_5 : tensor<?x?x4096xf32>, tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?x4096xf32>) {
          ^bb0(%in: f32, %in_6: f32, %in_7: f32, %out: f32):
            %38 = arith.mulf %in_6, %in_7 : f32
            %39 = arith.addf %in, %38 : f32
            linalg.yield %39 : f32
          } -> tensor<?x?x4096xf32>
          affine.yield %27, %32, %37 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>
        }
        // l_i
        %extracted_slice = tensor.extract_slice %15#1[0, 0] [%0, %1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        // acc /= l_i
        %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%15#2, %extracted_slice : tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?x4096xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %18 = arith.divf %in, %in_4 : f32
          linalg.yield %18 : f32
        } -> tensor<?x?x4096xf32>
        %17 = loom.subview %arg3[%10, %11, 0] [%0, %1, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy_from_tensor %16, %17 : tensor<?x?x4096xf32>, memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
      }
      return
    }
  }
}
