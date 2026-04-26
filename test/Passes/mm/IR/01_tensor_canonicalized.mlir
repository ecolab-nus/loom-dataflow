module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @_matmul(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %0 = loom.sym @tile_m {upper_bound = 4096 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 4096 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 256 : index} : index
    %3 = arith.ceildivui %c4096, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %6 = arith.ceildivui %c256, %2 : index
      %7 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %8 = scf.for %arg5 = %c0 to %6 step %c1 iter_args(%arg6 = %7) -> (tensor<?x?xf16>) {
        %14 = arith.muli %arg3, %0 : index
        %15 = arith.muli %arg5, %2 : index
        %subview_0 = memref.subview %arg0[%14, %15] [%0, %2] [1, 1] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %16 = bufferization.to_tensor %subview_0 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %17 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %18 = loom.sync ins(%16 : tensor<?x?xf16>) outs(%17 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %19 = arith.muli %arg4, %1 : index
        %subview_1 = memref.subview %arg1[%15, %19] [%2, %1] [1, 1] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        %20 = bufferization.to_tensor %subview_1 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
        %21 = tensor.empty(%2, %1) : tensor<?x?xf16>
        %22 = loom.sync ins(%20 : tensor<?x?xf16>) outs(%21 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %23 = linalg.fill ins(%cst : f16) outs(%5 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %24 = linalg.matmul ins(%18, %22 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg6, %24 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%5 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %26 = arith.addf %in, %in_2 : f16
          linalg.yield %26 : f16
        } -> tensor<?x?xf16>
        scf.yield %25 : tensor<?x?xf16>
      }
      %9 = arith.muli %arg3, %0 : index
      %10 = arith.muli %arg4, %1 : index
      %subview = memref.subview %arg2[%9, %10] [%0, %1] [1, 1] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %11 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %12 = loom.sync ins(%8 : tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %13 = bufferization.to_buffer %12 : tensor<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      memref.copy %13, %subview : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
    }
    return
  }
}
