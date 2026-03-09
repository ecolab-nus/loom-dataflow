// affine_maps
#map = affine_map<()[s0] -> (512 ceildiv s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module {
  module {
    func.func @matmul(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
      %cst = arith.constant 0.000000e+00 : f16
      // Manually added constraint space access
      %0 = loom.get_symbolic_block_size @constraints::@BM : index
      %1 = loom.get_symbolic_block_size @constraints::@BN : index
      %2 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3, %arg4) = (0, 0) to (4096 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = tensor.empty(%0, %1) : tensor<?x?xf16>
        %4 = linalg.fill ins(%cst : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %5 = affine.for %arg5 = 0 to #map()[%2] iter_args(%arg6 = %4) -> (tensor<?x?xf16>) {
          %9 = arith.muli %arg3, %0 : index
          %10 = arith.muli %arg5, %2 : index
          %subview_0 = memref.subview %arg0[%9, %10] [%0, %2] [1, 1] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
          %11 = bufferization.to_tensor %subview_0 : memref<?x?xf16, strided<[512, 1], offset: ?>> to tensor<?x?xf16>
          %12 = arith.muli %arg4, %1 : index
          %subview_1 = memref.subview %arg1[%10, %12] [%2, %1] [1, 1] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
          %13 = bufferization.to_tensor %subview_1 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
          %14 = linalg.matmul ins(%11, %13 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%4 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %15 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6, %14 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_2: f16, %out: f16):
            %16 = arith.addf %in, %in_2 : f16
            linalg.yield %16 : f16
          } -> tensor<?x?xf16>
          affine.yield %15 : tensor<?x?xf16>
        }
        %6 = arith.muli %arg3, %0 : index
        %7 = arith.muli %arg4, %1 : index
        %subview = memref.subview %arg2[%6, %7] [%0, %1] [1, 1] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        %8 = bufferization.to_buffer %5 : tensor<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        memref.copy %8, %subview : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      }
      return
    }
  }
} 