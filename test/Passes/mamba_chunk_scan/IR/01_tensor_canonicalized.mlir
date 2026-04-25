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
      %16 = bufferization.to_tensor %subview : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
      %17 = tensor.empty(%0) : tensor<?xf16>
      %18 = loom.sync ins(%16 : tensor<?xf16>) outs(%17 : tensor<?xf16>) -> tensor<?xf16>
      %19 = tensor.empty(%0) : tensor<?x32xf16>
      %20 = loom.broadcast ins(%18 : tensor<?xf16>) outs(%19 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
      %21 = arith.muli %14, %c256 : index
      %22 = arith.addi %15, %21 : index
      %23 = arith.divui %13, %c64 : index
      %subview_0 = memref.subview %arg4[%12, %22, %23, 0] [1, %0, 1, 64] [1, 1, 1, 1] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      %24 = bufferization.to_tensor %subview_0 : memref<?x64xf16, strided<[64, 1], offset: ?>> to tensor<?x64xf16>
      %25 = tensor.empty(%0) : tensor<?x64xf16>
      %26 = loom.sync ins(%24 : tensor<?x64xf16>) outs(%25 : tensor<?x64xf16>) -> tensor<?x64xf16>
      %27 = arith.muli %arg10, %1 : index
      %subview_1 = memref.subview %arg5[%12, %14, %13, 0, %27] [1, 1, 1, 64, %1] [1, 1, 1, 1, 1] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
      %28 = bufferization.to_tensor %subview_1 : memref<64x?xf16, strided<[64, 1], offset: ?>> to tensor<64x?xf16>
      %29 = tensor.empty(%1) : tensor<64x?xf16>
      %30 = loom.sync ins(%28 : tensor<64x?xf16>) outs(%29 : tensor<64x?xf16>) -> tensor<64x?xf16>
      %31 = linalg.fill ins(%cst : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %32 = linalg.matmul ins(%26, %30 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%31 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %33 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32, %20 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_5: f16, %out: f16):
        %48 = math.exp %in_5 : f16
        %49 = arith.mulf %in, %48 : f16
        linalg.yield %49 : f16
      } -> tensor<?x?xf16>
      %34 = arith.addi %arg9, %c1 : index
      %35 = arith.muli %34, %0 : index
      %36 = arith.ceildivui %35, %2 : index
      %37 = scf.for %arg13 = %c0 to %36 step %c1 iter_args(%arg14 = %33) -> (tensor<?x?xf16>) {
        %48 = arith.muli %arg13, %2 : index
        %subview_5 = memref.subview %arg0[%12, %14, %23, %15, %48] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        %49 = bufferization.to_tensor %subview_5 : memref<?x?xf16, strided<[256, 1], offset: ?>> to tensor<?x?xf16>
        %50 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %51 = loom.sync ins(%49 : tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %subview_6 = memref.subview %arg1[%12, %13, %14, %48] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %52 = bufferization.to_tensor %subview_6 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %53 = tensor.empty(%2) : tensor<?xf16>
        %54 = loom.sync ins(%52 : tensor<?xf16>) outs(%53 : tensor<?xf16>) -> tensor<?xf16>
        %55 = loom.broadcast ins(%18 : tensor<?xf16>) outs(%19 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
        %56 = tensor.empty(%2) : tensor<32x?xf16>
        %57 = loom.broadcast ins(%54 : tensor<?xf16>) outs(%56 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        %58 = tensor.empty(%0, %2) : tensor<?x?xf16>
        %subview_7 = memref.subview %arg2[%12, %13, %14, %48] [1, 1, 1, %2] [1, 1, 1, 1] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        %59 = bufferization.to_tensor %subview_7 : memref<?xf16, strided<[1], offset: ?>> to tensor<?xf16>
        %60 = tensor.empty(%2) : tensor<?xf16>
        %61 = loom.sync ins(%59 : tensor<?xf16>) outs(%60 : tensor<?xf16>) -> tensor<?xf16>
        %62 = loom.broadcast ins(%61 : tensor<?xf16>) outs(%56 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%51, %55, %57, %62 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%58 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
          %71 = arith.subf %in_9, %in_10 : f16
          %72 = math.exp %71 : f16
          %73 = arith.mulf %in, %72 : f16
          %74 = arith.mulf %73, %in_11 : f16
          linalg.yield %74 : f16
        } -> tensor<?x?xf16>
        %64 = arith.addi %48, %21 : index
        %subview_8 = memref.subview %arg3[%12, %64, %13, %27] [1, %2, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        %65 = bufferization.to_tensor %subview_8 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
        %66 = tensor.empty(%2, %1) : tensor<?x?xf16>
        %67 = loom.sync ins(%65 : tensor<?x?xf16>) outs(%66 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %68 = linalg.fill ins(%cst : f16) outs(%11 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %69 = linalg.matmul ins(%63, %67 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%68 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %69 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%11 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_9: f16, %out: f16):
          %71 = arith.addf %in, %in_9 : f16
          linalg.yield %71 : f16
        } -> tensor<?x?xf16>
        scf.yield %70 : tensor<?x?xf16>
      }
      %subview_2 = memref.subview %arg6[%13] [1] [1] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      %38 = bufferization.to_tensor %subview_2 : memref<f16, strided<[], offset: ?>> to tensor<f16>
      %39 = tensor.empty() : tensor<f16>
      %40 = loom.sync ins(%38 : tensor<f16>) outs(%39 : tensor<f16>) -> tensor<f16>
      %subview_3 = memref.subview %arg3[%12, %22, %13, %27] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %41 = bufferization.to_tensor %subview_3 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to tensor<?x?xf16>
      %42 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %43 = loom.sync ins(%41 : tensor<?x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%37, %43, %40 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%11 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
        %48 = arith.mulf %in_5, %in_6 : f16
        %49 = arith.addf %in, %48 : f16
        linalg.yield %49 : f16
      } -> tensor<?x?xf16>
      %subview_4 = memref.subview %arg7[%12, %22, %13, %27] [1, %0, 1, %1] [1, 1, 1, 1] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %45 = tensor.empty(%0, %1) : tensor<?x?xf16>
      %46 = loom.sync ins(%44 : tensor<?x?xf16>) outs(%45 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %47 = bufferization.to_buffer %46 : tensor<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      memref.copy %47, %subview_4 : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
    }
    return
  }
}
