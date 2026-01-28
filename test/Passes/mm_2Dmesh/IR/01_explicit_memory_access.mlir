module {
  module {
    loom.constraint_space @constraints {
      %0 = loom.symbolic_var "BM" : index
      %1 = loom.symbolic_var "BN" : index
      %2 = loom.symbolic_var "BK" : index
      loom.range %0[0, 1024]
      loom.range %1[0, 1024]
      loom.range %2[0, 1024]
    }
    func.func @matmul(%arg0: memref<128x128xf32>, %arg1: memref<128x256xf32>, %arg2: memref<128x256xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = loom.get_symbolic_block_size @constraints::@BM : index
      %1 = loom.get_symbolic_block_size @constraints::@BN : index
      %2 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3, %arg4) = (0, 0) to (128 ceildiv symbol(%0), 256 ceildiv symbol(%1)) {
        %3 = loom.allocc(%0, %1) on @L1 : !loom.buffer_token
        %4 = loom.init_tensor %3(%0, %1) : !loom.buffer_token -> tensor<?x?xf32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %6 = affine.for %arg5 = 0 to affine_map<()[s0] -> (128 ceildiv s0)>()[%2] iter_args(%arg6 = %5) -> (tensor<?x?xf32>) {
          %10 = arith.muli %arg5, %2 : index
          %11 = loom.view %arg0[%0, %10] [%0, %2] [1, 1] : memref<128x128xf32> -> !loom.view
          %12 = loom.allocc(%0, %2) on @L1 : !loom.buffer_token
          %13 = loom.copy_to_tensor %11, %12, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
          %14 = loom.view %arg1[%10, %1] [%2, %1] [1, 1] : memref<128x256xf32> -> !loom.view
          %15 = loom.allocc(%2, %1) on @L1 : !loom.buffer_token
          %16 = loom.copy_to_tensor %14, %15, interconnect : [], broadcast : [1, 1] : !loom.view, !loom.buffer_token -> tensor<?x?xf32>
          %17 = linalg.matmul ins(%13, %16 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg6, %17 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_0: f32, %out: f32):
            %19 = arith.addf %in, %in_0 : f32
            linalg.yield %19 : f32
          } -> tensor<?x?xf32>
          affine.yield %18 : tensor<?x?xf32>
        }
        %7 = arith.muli %arg3, %0 : index
        %8 = arith.muli %arg4, %1 : index
        %9 = loom.view %arg2[%7, %8] [%0, %1] [1, 1] : memref<128x256xf32> -> !loom.view
        loom.copy_from_tensor %6, %9 : tensor<?x?xf32>, !loom.view
      }
      return
    }
  }
}
