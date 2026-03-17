module attributes {loom.block_size_0 = -1 : index, loom.block_size_1 = -1 : index, loom.block_size_2 = -1 : index} {
  func.func @matmul(%x: memref<4096x512xf32>, %y: memref<512x4096xf32>, %out: memref<4096x4096xf32>) {
    %block_size_0 = loom.sym @block_size_0 : index
    %block_size_1 = loom.sym @block_size_1 : index
    %block_size_2 = loom.sym @block_size_2 : index
    %trip_count0 = affine.apply affine_map<()[s0] -> (512 ceildiv s0)>()[%block_size_2]
    %trip_count1 = affine.apply affine_map<()[s0] -> (4096 ceildiv s0)>()[%block_size_0]
    %trip_count2 = affine.apply affine_map<()[s0] -> (4096 ceildiv s0)>()[%block_size_1]
    affine.parallel (%iv_block_0, %iv_block_1) = (0, 0) to (%trip_count1, %trip_count2) {
      %empty3 = tensor.empty(%block_size_0, %block_size_1) : tensor<?x?xf32>
      %fill_val4 = arith.constant 0.0 : f32
      %filled5 = linalg.fill ins(%fill_val4 : f32) outs(%empty3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %x_size16 = arith.constant 512 : index
      %for_result_07 = affine.for %iv_block_2 = 0 to %trip_count0 iter_args(%acc_iter0 = %filled5) -> (tensor<?x?xf32>) {
        %offset8 = arith.muli %iv_block_0, %block_size_0 : index
        %offset9 = arith.muli %iv_block_2, %block_size_2 : index
        %subview10 = memref.subview %x[%offset8, %offset9][%block_size_0, %block_size_2][1, 1] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
        %tile11 = bufferization.to_tensor %subview10 : memref<?x?xf32, strided<[512, 1], offset: ?>> to tensor<?x?xf32>
        %offset12 = arith.muli %iv_block_2, %block_size_2 : index
        %offset13 = arith.muli %iv_block_1, %block_size_1 : index
        %subview14 = memref.subview %y[%offset12, %offset13][%block_size_2, %block_size_1][1, 1] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
        %tile15 = bufferization.to_tensor %subview14 : memref<?x?xf32, strided<[4096, 1], offset: ?>> to tensor<?x?xf32>
        %t18 = arith.constant 0.000000e+00 : f32
        %t19 = arith.cmpi eq, %block_size_2, %block_size_2 : index
        cf.assert %t19, "mismatching contracting dimension for torch.aten.mm"
        %t20 = tensor.empty(%block_size_0, %block_size_1) : tensor<?x?xf32>
        %t21 = linalg.fill ins(%t18 : f32) outs(%t20 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %t22 = linalg.matmul ins(%tile11, %tile15 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%t21 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %t23 = arith.cmpi eq, %block_size_0, %block_size_0 : index
        cf.assert %t23, "mismatched size for broadcast"
        %t24 = arith.cmpi eq, %block_size_1, %block_size_1 : index
        cf.assert %t24, "mismatched size for broadcast"
        %t25 = tensor.empty(%block_size_0, %block_size_1) : tensor<?x?xf32>
        %t26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%acc_iter0, %t22 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%t25 : tensor<?x?xf32>) {
        ^bb0(%blk_arg27: f32, %blk_arg28: f32, %blk_arg29: f32):
        %t30 = arith.addf %blk_arg27, %blk_arg28 : f32
        linalg.yield %t30 : f32
        } -> tensor<?x?xf32>
        affine.yield %t26 : tensor<?x?xf32>
      }
      %offset31 = arith.muli %iv_block_0, %block_size_0 : index
      %offset32 = arith.muli %iv_block_1, %block_size_1 : index
      %subview33 = memref.subview %out[%offset31, %offset32][%block_size_0, %block_size_1][1, 1] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      %value_memref34 = bufferization.to_buffer %for_result_07 : tensor<?x?xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      memref.copy %value_memref34, %subview33 : memref<?x?xf32, strided<[4096, 1], offset: ?>> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
      affine.yield
    }
    return
  }
}
