module {
  module {
    loom.constraint_space @constraints {
      %0 = loom.symbolic_var "M" : index
      %1 = loom.symbolic_var "N" : index
      %2 = loom.symbolic_var "K" : index
      loom.range %0[0, 512]
      loom.align %0 by 32
      loom.range %1[0, 512]
      loom.align %1 by 32
      loom.range %2[0, 512]
      loom.align %2 by 32
    }
    func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
      affine.parallel (%arg9, %arg10) = (0, 0) to (%arg3, %arg4) {
        %c512 = arith.constant 512 : index
        %c512_0 = arith.constant 512 : index
        %c512_1 = arith.constant 512 : index
        %c1 = arith.constant 1 : index
        %0 = loom.get_symbolic_block_size @constraints::@M : index
        %1 = loom.get_symbolic_block_size @constraints::@N : index
        %2 = loom.get_symbolic_block_size @constraints::@K : index
        %3 = arith.ceildivsi %c512_1, %2 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c8 = arith.constant 8 : index
        %c512_2 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
        %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %6 = arith.muli %arg9, %0 : index
        %7 = arith.muli %arg10, %1 : index
        %8 = scf.for %arg11 = %c0 to %3 step %c1 iter_args(%arg12 = %5) -> (tensor<?x?xf32>) {
          %12 = arith.muli %arg11, %2 : index
          %13 = arith.muli %6, %c512_2 : index
          %14 = arith.addi %13, %12 : index
          %15 = loom.reinterpret_cast %arg0 to offset : [%14], sizes : [%0, %2], strides : [%c512_1, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %16 = loom.alloc(%0, %2) on @L1 : memref<?x?xf32>
          loom.copy %15, %16, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
          %17 = bufferization.to_tensor %16 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
          %18 = arith.muli %12, %c512_2 : index
          %19 = arith.addi %18, %7 : index
          %20 = loom.reinterpret_cast %arg1 to offset : [%19], sizes : [%2, %1], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
          %21 = loom.alloc(%2, %1) on @L1 : memref<?x?xf32>
          loom.copy %20, %21, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
          %22 = bufferization.to_tensor %21 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
          %23 = linalg.matmul ins(%17, %22 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg12, %23 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg12 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_3: f32, %out: f32):
            %25 = arith.addf %in, %in_3 : f32
            linalg.yield %25 : f32
          } -> tensor<?x?xf32>
          scf.yield %24 : tensor<?x?xf32>
        }
        %9 = arith.muli %6, %c512_2 : index
        %10 = arith.addi %9, %7 : index
        %11 = loom.reinterpret_cast %arg2 to offset : [%10], sizes : [%0, %1], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        bufferization.materialize_in_destination %8 in writable %11 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
      }
      return
    }
  }
}
