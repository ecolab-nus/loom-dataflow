module attributes {loom.block_k = 64 : index, loom.block_m = 64 : index, loom.block_n = 64 : index} {
  func.func @matmul_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index) {
    affine.parallel (%arg9, %arg10) = (0, 0) to (%arg3, %arg4) {
      %c512 = arith.constant 512 : index
      %c512_0 = arith.constant 512 : index
      %c512_1 = arith.constant 512 : index
      %c1 = arith.constant 1 : index
      %0 = loom.get_module_attribute "loom.block_m" : index
      %1 = loom.get_module_attribute "loom.block_n" : index
      %2 = loom.get_module_attribute "loom.block_k" : index
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
        %alloc = memref.alloc(%0, %2) : memref<?x?xf32>
        loom.copy %15, %alloc, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
        %16 = bufferization.to_tensor %alloc restrict writable : memref<?x?xf32> to tensor<?x?xf32>
        %17 = arith.muli %12, %c512_2 : index
        %18 = arith.addi %17, %7 : index
        %19 = loom.reinterpret_cast %arg1 to offset : [%18], sizes : [%2, %1], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
        %alloc_3 = memref.alloc(%2, %1) : memref<?x?xf32>
        loom.copy %19, %alloc_3, interconnect : [], broadcast : [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<?x?xf32>
        %20 = bufferization.to_tensor %alloc_3 restrict writable : memref<?x?xf32> to tensor<?x?xf32>
        %21 = linalg.matmul ins(%16, %20 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg12, %21 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg12 : tensor<?x?xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %23 = arith.addf %in, %in_4 : f32
          linalg.yield %23 : f32
        } -> tensor<?x?xf32>
        scf.yield %22 : tensor<?x?xf32>
      }
      %9 = arith.muli %6, %c512_2 : index
      %10 = arith.addi %9, %7 : index
      %11 = loom.reinterpret_cast %arg2 to offset : [%10], sizes : [%0, %1], strides : [%c512_0, %c1], reuse : [seq = false, spat = false, temp = false] : memref<*xf32> to memref<?x?xf32, strided<[?, ?], offset: ?>>
      bufferization.materialize_in_destination %8 in writable %11 : (tensor<?x?xf32>, memref<?x?xf32, strided<[?, ?], offset: ?>>) -> ()
    }
    return
  }
}
