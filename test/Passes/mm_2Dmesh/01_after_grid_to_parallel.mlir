module attributes {loom.block_k = 64 : index, loom.block_m = 64 : index, loom.block_n = 64 : index} {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9, %arg10) = (0, 0) to (%arg3, %arg4) {
      %c512 = arith.constant 512 : index
      %c512_0 = arith.constant 512 : index
      %c512_1 = arith.constant 512 : index
      %c1 = arith.constant 1 : index
      %0 = loom.get_block_size "m" : index
      %1 = loom.get_block_size "n" : index
      %2 = loom.get_block_size "k" : index
      %cst = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c512_2 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %5 = arith.muli %arg9, %0 : index
      %6 = arith.muli %arg10, %1 : index
      %7 = scf.for %arg11 = %c0 to %c8 step %c1 iter_args(%arg12 = %4) -> (tensor<?x?xf32>) {
        %11 = arith.muli %arg11, %2 : index
        %12 = arith.muli %5, %c512_2 : index
        %13 = arith.addi %12, %11 : index
        %14 = loom.reinterpret_cast %arg0 to offset : [%13], sizes : [%0, %2], strides : [%c512_1, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %alloc = memref.alloc(%0, %2) : memref<?x?xf32>
        loom.copy %14, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
        %15 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
        %16 = arith.muli %11, %c512_2 : index
        %17 = arith.addi %16, %6 : index
        %18 = loom.reinterpret_cast %arg1 to offset : [%17], sizes : [%2, %1], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %alloc_3 = memref.alloc(%2, %1) : memref<?x?xf32>
        loom.copy %18, %alloc_3, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
        %19 = bufferization.to_tensor %alloc_3 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
        %20 = linalg.matmul ins(%15, %19 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg12, %20 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg12 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %22 = arith.addf %in, %in_4 : f32
          linalg.yield %22 : f32
        } -> tensor<?x?xf32>
        scf.yield %21 : tensor<?x?xf32>
      }
      %8 = arith.muli %5, %c512_2 : index
      %9 = arith.addi %8, %6 : index
      %10 = loom.reinterpret_cast %arg2 to offset : [%9], sizes : [%0, %1], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      bufferization.materialize_in_destination %7 in writable %10 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
    }
    return
  }
}
