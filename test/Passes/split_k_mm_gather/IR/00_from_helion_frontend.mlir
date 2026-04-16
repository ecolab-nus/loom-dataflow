module attributes {loom.tile_m = {upper_bound = 512 : index, is_reduction = false}, loom.tile_n = {upper_bound = 512 : index, is_reduction = false}, loom.tile_k = {upper_bound = 4096 : index, is_reduction = false}} {
  func.func @split_k_matmul_gather(%out: memref<512x512xf16>, %a: memref<512x4096xf16>, %b: memref<4096x512xf16>) {
    %tile_m = "loom.sym"() {symbol_ref = @tile_m, upper_bound = 512 : index, is_reduction = false} : () -> index
    %tile_n = "loom.sym"() {symbol_ref = @tile_n, upper_bound = 512 : index, is_reduction = false} : () -> index
    %tile_k = "loom.sym"() {symbol_ref = @tile_k, upper_bound = 4096 : index, is_reduction = false} : () -> index
    %loop_extent0 = arith.constant 512 : index
    %trip_count1 = arith.ceildivui %loop_extent0, %tile_m : index
    %loop_extent2 = arith.constant 512 : index
    %trip_count3 = arith.ceildivui %loop_extent2, %tile_n : index
    %loop_extent4 = arith.constant 4096 : index
    %trip_count5 = arith.ceildivui %loop_extent4, %tile_k : index
    affine.parallel (%iv_block_0, %iv_block_1, %iv_block_2) = (0, 0, 0) to (%trip_count1, %trip_count3, %trip_count5) {
      %offset6 = arith.muli %iv_block_0, %tile_m : index
      %offset7 = arith.muli %iv_block_2, %tile_k : index
      %subview8 = memref.subview %a[%offset6, %offset7][%tile_m, %tile_k][1, 1] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %tile9 = bufferization.to_tensor %subview8 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
      %offset10 = arith.muli %iv_block_2, %tile_k : index
      %offset11 = arith.muli %iv_block_1, %tile_n : index
      %subview12 = memref.subview %b[%offset10, %offset11][%tile_k, %tile_n][1, 1] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
      %tile13 = bufferization.to_tensor %subview12 : memref<?x?xf16, strided<[512, 1], offset: ?>> to tensor<?x?xf16>
      %t16 = arith.constant 0.000000e+00 : f16
      %t17 = arith.cmpi eq, %tile_k, %tile_k : index
      cf.assert %t17, "mismatching contracting dimension for torch.aten.mm"
      %t18 = tensor.empty(%tile_m, %tile_n) : tensor<?x?xf16>
      %t19 = linalg.fill ins(%t16 : f16) outs(%t18 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %t20 = linalg.matmul ins(%tile9, %tile13 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%t19 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %t21 = tensor.empty(%tile_m, %tile_n) : tensor<?x?xf16>
      %loop_extent22 = arith.constant 4096 : index
      %trip_count23 = arith.ceildivui %loop_extent22, %tile_k : index
      %gather_out24 = tensor.empty(%trip_count23, %tile_m, %tile_n) : tensor<?x?x?xf16>
      %gathered25 = "loom.gather"(%t20, %gather_out24, %iv_block_2) {operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0>} : (tensor<?x?xf16>, tensor<?x?x?xf16>, index) -> tensor<?x?x?xf16>
      %cmp_rhs26 = arith.constant 0 : index
      %cmp27 = arith.cmpi eq, %iv_block_2, %cmp_rhs26 : index
      scf.if %cmp27 {
        %t28 = arith.constant 0.000000e+00 : f16
        %t31 = tensor.empty(%tile_m, %tile_n) : tensor<?x?xf16>
        %t32 = linalg.fill ins(%t28 : f16) outs(%t31 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %t33 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%gathered25 : tensor<?x?x?xf16>) outs(%t32 : tensor<?x?xf16>) {
        ^bb0(%blk_arg34: f16, %blk_arg35: f16):
        %t36 = arith.addf %blk_arg34, %blk_arg35 : f16
        linalg.yield %t36 : f16
        } -> tensor<?x?xf16>
        %offset37 = arith.muli %iv_block_0, %tile_m : index
        %offset38 = arith.muli %iv_block_1, %tile_n : index
        %subview39 = memref.subview %out[%offset37, %offset38][%tile_m, %tile_n][1, 1] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        %value_memref40 = bufferization.to_buffer %t33 : tensor<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        memref.copy %value_memref40, %subview39 : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
      }
      affine.yield
    }
    return
  }
}