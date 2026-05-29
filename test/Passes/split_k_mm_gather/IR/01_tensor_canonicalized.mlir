module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
  func.func @split_k_matmul_gather(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c512 = arith.constant 512 : index
    %0 = loom.sym @tile_m {upper_bound = 512 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 512 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 4096 : index} : index
    %3 = arith.ceildivui %c512, %0 : index
    %4 = arith.ceildivui %c512, %1 : index
    %5 = arith.ceildivui %c4096, %2 : index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%3), symbol(%4), symbol(%5)) {
      %6 = arith.muli %arg3, %0 : index
      %7 = arith.muli %arg5, %2 : index
      %subview = memref.subview %arg0[%6, %7] [%0, %2] [1, 1] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %8 = bufferization.to_tensor %subview : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
      %9 = tensor.empty(%0, %2) : tensor<?x?xf16>
      %10 = loom.sync ins(%8 : tensor<?x?xf16>) outs(%9 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %11 = arith.muli %arg5, %2 : index
      %12 = arith.muli %arg4, %1 : index
      %subview_0 = memref.subview %arg1[%11, %12] [%2, %1] [1, 1] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
      %13 = bufferization.to_tensor %subview_0 : memref<?x?xf16, strided<[512, 1], offset: ?>> to tensor<?x?xf16>
      %14 = tensor.empty(%2, %1) : tensor<?x?xf16>
      %15 = loom.sync ins(%13 : tensor<?x?xf16>) outs(%14 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %16 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %17 = linalg.fill ins(%cst : f16) outs(%16 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %18 = linalg.matmul ins(%10, %15 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%17 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %19 = arith.ceildivui %c4096, %2 : index
      %20 = tensor.empty(%19, %0, %1) : tensor<?x?x?xf16>
      %21 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %22 = loom.sync ins(%18 : tensor<?x?xf16>) outs(%21 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %23 = loom.gather ins(%22 : tensor<?x?xf16>) outs(%20 : tensor<?x?x?xf16>) across(%arg5 : index) -> tensor<?x?x?xf16>
      %24 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %24 {
        %25 = tensor.empty(%0, %1) : tensor<?x?xf16>
        %26 = linalg.fill ins(%cst : f16) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %27 = tensor.empty(%19, %0, %1) : tensor<?x?x?xf16>
        %28 = loom.sync ins(%23 : tensor<?x?x?xf16>) outs(%27 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %29 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%28 : tensor<?x?x?xf16>) outs(%26 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %35 = arith.addf %in, %out : f16
          linalg.yield %35 : f16
        } -> tensor<?x?xf16>
        %30 = arith.muli %arg3, %0 : index
        %31 = arith.muli %arg4, %1 : index
        %subview_1 = memref.subview %arg2[%30, %31] [%0, %1] [1, 1] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        %32 = tensor.empty(%0, %1) : tensor<?x?xf16>
        %33 = loom.sync ins(%29 : tensor<?x?xf16>) outs(%32 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %34 = bufferization.to_buffer %33 : tensor<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        memref.copy %34, %subview_1 : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16, strided<[512, 1], offset: ?>>
      }
    }
    return
  }
}
