module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_h {upper_bound = 64 : index} : index
    %4 = loom.sym @tile_b {upper_bound = 2 : index} : index
    %5 = loom.sym @tile_c {upper_bound = 8 : index} : index
    %6 = arith.ceildivui %c64, %3 : index
    %7 = arith.ceildivui %c256, %0 : index
    %8 = arith.ceildivui %c64, %1 : index
    %9 = arith.ceildivui %c2, %4 : index
    %10 = arith.ceildivui %c8, %5 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%6), symbol(%7), symbol(%8), symbol(%9), symbol(%10)) {
      %11 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %12 = arith.muli %arg11, %4 : index
      %13 = arith.muli %arg8, %3 : index
      %14 = arith.muli %arg12, %5 : index
      %15 = arith.muli %arg9, %0 : index
      %subview = memref.subview %arg1[%12, %13, %14, %15] [1, 1, 1, %0] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      %16 = loom.bufferize_to_tensor %subview[%0] : memref<?xf16, strided<[1], offset: ?>> -> tensor<?xf16>
      %17 = tensor.empty(%0) : tensor<?x32xf16>
      %18 = loom.broadcast ins(%16 : tensor<?xf16>) outs(%17 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
      %19 = arith.muli %14, %c256 : index
      %20 = arith.addi %15, %19 : index
      %21 = arith.divui %13, %c64 : index
      %subview_0 = memref.subview %arg4[%12, %20, %21, 0] [1, %0, 1, 64] [1, 1, 1, 1] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %22 = loom.bufferize_to_tensor %subview_0[%0, 64] : memref<?x64xf16, strided<[64, 1], offset: ?>> -> tensor<?x64xf16>
      %23 = arith.muli %arg10, %1 : index
      %subview_1 = memref.subview %arg5[%12, %14, %13, 0, %23] [1, 1, 1, 64, %1] [1, 1, 1, 1, 1] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
      %24 = loom.bufferize_to_tensor %subview_1[64, %1] : memref<64x?xf16, strided<[64, 1], offset: ?>> -> tensor<64x?xf16>
      %25 = linalg.fill ins(%cst : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %26 = linalg.matmul ins(%22, %24 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%25 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %27 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%26, %18 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_5: f16, %out: f16):
        %36 = math.exp %in_5 : f16
        %37 = arith.mulf %in, %36 : f16
        linalg.yield %37 : f16
      } -> tensor<?x?xf16>
      %28 = arith.addi %arg9, %c1 : index
      %29 = arith.muli %28, %0 : index
      %30 = arith.ceildivui %29, %2 : index
      %31 = scf.for %arg13 = %c0 to %30 step %c1 iter_args(%arg14 = %27) -> (tensor<?x?xf16>) {
        %36 = arith.muli %arg13, %2 : index
        %subview_5 = memref.subview %arg0[%12, %14, %21, %15, %36] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %37 = loom.bufferize_to_tensor %subview_5[%0, %2] : memref<?x?xf16, strided<[256, 1], offset: ?>> -> tensor<?x?xf16>
        %subview_6 = memref.subview %arg1[%12, %13, %14, %36] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %38 = loom.bufferize_to_tensor %subview_6[%2] : memref<?xf16, strided<[1], offset: ?>> -> tensor<?xf16>
        %39 = loom.broadcast ins(%16 : tensor<?xf16>) outs(%17 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
        %40 = tensor.empty(%2) : tensor<32x?xf16>
        %41 = loom.broadcast ins(%38 : tensor<?xf16>) outs(%40 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        %42 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %subview_7 = memref.subview %arg2[%12, %13, %14, %36] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %43 = loom.bufferize_to_tensor %subview_7[%2] : memref<?xf16, strided<[1], offset: ?>> -> tensor<?xf16>
        %44 = loom.broadcast ins(%43 : tensor<?xf16>) outs(%40 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%37, %39, %41, %44 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
          %51 = arith.subf %in_9, %in_10 : f16
          %52 = math.exp %51 : f16
          %53 = arith.mulf %in, %52 : f16
          %54 = arith.mulf %53, %in_11 : f16
          linalg.yield %54 : f16
        } -> tensor<?x?xf16>
        %46 = arith.addi %36, %19 : index
        %subview_8 = memref.subview %arg3[%12, %46, %13, %23] [1, %2, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        %47 = loom.bufferize_to_tensor %subview_8[%2, %1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> -> tensor<?x?xf16>
        %48 = linalg.fill ins(%cst : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %49 = linalg.matmul ins(%45, %47 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %49 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_9: f16, %out: f16):
          %51 = arith.addf %in, %in_9 : f16
          linalg.yield %51 : f16
        } -> tensor<?x?xf16>
        scf.yield %50 : tensor<?x?xf16>
      }
      %subview_2 = memref.subview %arg6[%13] [1] [1] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      %32 = loom.bufferize_to_tensor %subview_2[] : memref<f16, strided<[], offset: ?>> -> tensor<f16>
      %subview_3 = memref.subview %arg3[%12, %20, %13, %23] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %33 = loom.bufferize_to_tensor %subview_3[%0, %1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> -> tensor<?x?xf16>
      %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %33, %32 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
        %36 = arith.mulf %in_5, %in_6 : f16
        %37 = arith.addf %in, %36 : f16
        linalg.yield %37 : f16
      } -> tensor<?x?xf16>
      %subview_4 = memref.subview %arg7[%12, %20, %13, %23] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %35 = loom.bufferize_to_memref %34 : tensor<?x?xf16> -> memref<?x?xf16, strided<[4096, 1], offset: ?>>
      memref.copy %35, %subview_4 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
    }
    return
  }
}
