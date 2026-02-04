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
    func.func @matmul(%arg0: memref<4096x512xf32>, %arg1: memref<512x4096xf32>, %arg2: memref<4096x4096xf32>) {
      %cst = arith.constant 0.000000e+00 : f32
      %0 = loom.get_symbolic_block_size @constraints::@BM : index
      %1 = loom.get_symbolic_block_size @constraints::@BN : index
      %2 = loom.get_symbolic_block_size @constraints::@BK : index
      affine.parallel (%arg3, %arg4) = (0, 0) to (4096 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = loom.alloc [%0, %1] on @L1 : !loom.buffer_token
        %4 = loom.init_tensor %3[%0, %1] : !loom.buffer_token -> tensor<?x?xf32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %6 = affine.for %arg5 = 0 to affine_map<()[s0] -> (512 ceildiv s0)>()[%2] iter_args(%arg6 = %5) -> (tensor<?x?xf32>) {
          %10 = arith.muli %arg3, %0 : index
          %11 = arith.muli %arg5, %2 : index
          %12 = loom.view %arg0[%10, %11] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf32> to memref<?x?xf32, strided<[512, 1], offset: ?>>
          %13 = loom.alloc [%0, %2] on @L1 : !loom.buffer_token
          %14 = loom.copy_to_tensor %12, %13, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[512, 1], offset: ?>>, !loom.buffer_token -> tensor<?x?xf32>
          %15 = arith.muli %arg4, %1 : index
          %16 = loom.view %arg1[%11, %15] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
          %17 = loom.alloc [%2, %1] on @L1 : !loom.buffer_token
          %18 = loom.copy_to_tensor %16, %17, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[4096, 1], offset: ?>>, !loom.buffer_token -> tensor<?x?xf32>
          %19 = linalg.matmul ins(%14, %18 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg6 : tensor<?x?xf32>) -> tensor<?x?xf32>
          affine.yield %19 : tensor<?x?xf32>
        }
        %7 = arith.muli %arg3, %0 : index
        %8 = arith.muli %arg4, %1 : index
        %9 = loom.view %arg2[%7, %8] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf32> to memref<?x?xf32, strided<[4096, 1], offset: ?>>
        loom.copy_from_tensor %6, %9 : tensor<?x?xf32>, memref<?x?xf32, strided<[4096, 1], offset: ?>>
      }
      return
    }
  }
}
