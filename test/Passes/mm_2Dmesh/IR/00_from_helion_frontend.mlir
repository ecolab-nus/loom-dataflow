// affine_maps
#map = affine_map<()[s0] -> (128 ceildiv s0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module {
  module {
    // Manually added constraint space
    loom.constraint_space @constraints {
      %bm = loom.symbolic_var "BM" : index
      %bn = loom.symbolic_var "BN" : index
      %bk = loom.symbolic_var "BK" : index

      loom.range %bm [0, 1024]
      loom.range %bn [0, 1024]
      loom.range %bk [0, 1024]
    }
    func.func @matmul(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      // Manually added constraint space access
      %0 = loom.get_symbolic_block_size @constraints::@BM : index
      %1 = loom.get_symbolic_block_size @constraints::@BN : index
      %2 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3, %arg4) = (0, 0) to (128 ceildiv symbol(%0), 256 ceildiv symbol(%1)) {
        %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %5 = affine.for %arg5 = 0 to #map()[%2] iter_args(%arg6 = %4) -> (tensor<?x?xf32>) {
          %9 = arith.muli %arg5, %2 : index
          %subview_0 = memref.subview %arg0[%0, %9] [%0, %2] [1, 1] : memref<128x128xf32> to memref<?x?xf32, strided<[128, 1], offset: ?>>
          %10 = bufferization.to_tensor %subview_0 : memref<?x?xf32, strided<[128, 1], offset: ?>> to tensor<?x?xf32>
          %subview_1 = memref.subview %arg1[%9, %1] [%2, %1] [1, 1] : memref<128x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
          %11 = bufferization.to_tensor %subview_1 : memref<?x?xf32, strided<[256, 1], offset: ?>> to tensor<?x?xf32>
          %12 = linalg.matmul ins(%10, %11 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %13 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg6, %12 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_2: f32, %out: f32):
          %14 = arith.addf %in, %in_2 : f32
          linalg.yield %14 : f32
          } -> tensor<?x?xf32>
          affine.yield %13 : tensor<?x?xf32>
        }
        %6 = arith.muli %arg3, %0 : index
        %7 = arith.muli %arg4, %1 : index
        %subview = memref.subview %arg2[%6, %7] [%0, %1] [1, 1] : memref<128x256xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        %8 = bufferization.to_buffer %5 : tensor<?x?xf32> to memref<?x?xf32, strided<[256, 1], offset: ?>>
        memref.copy %8, %subview : memref<?x?xf32, strided<[256, 1], offset: ?>> to memref<?x?xf32, strided<[256, 1], offset: ?>>
      }
      return
    }
  }
} 