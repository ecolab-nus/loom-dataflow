module {
  module {
    func.func @attention(%arg0: memref<1x128x4096xf16>, %arg1: memref<1x4096x128xf16>, %arg2: memref<1x4096x128xf16>, %arg3: memref<1x4096x128xf16>) {
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 0.127517432 : f32
      %cst_1 = arith.constant 0.000000e+00 : f16
      %cst_2 = arith.constant 1.000000e+00 : f16
      %cst_3 = arith.constant 0xFC00 : f16
      %0 = loom.get_symbolic_block_size @constraints::@BB : index
      %1 = loom.get_symbolic_block_size @constraints::@BM : index
      %2 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4, %arg5) = (0, 0) to (1 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = tensor.empty(%0, %1) : tensor<?x?xf16>
        %4 = tensor.empty(%0, %1) : tensor<?x?x128xf16>
        %5 = arith.muli %arg4, %0 : index
        %6 = arith.muli %arg5, %1 : index
        %subview = memref.subview %arg2[%5, %6, 0] [%0, %1, 128] [1, 1, 1] : memref<1x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %7 = bufferization.to_tensor %subview : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
        %8 = linalg.fill ins(%cst_1 : f16) outs(%4 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %9 = linalg.fill ins(%cst_2 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %10 = linalg.fill ins(%cst_3 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %11:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %10, %arg8 = %9, %arg9 = %8) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
          %14 = arith.muli %arg6, %2 : index
          %subview_5 = memref.subview %arg0[%5, 0, %14] [%0, 128, %2] [1, 1, 1] : memref<1x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
          %15 = bufferization.to_tensor %subview_5 : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf16>
          %16 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf16>
          %17 = linalg.fill ins(%cst_1 : f16) outs(%16 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
          %18 = linalg.batch_matmul ins(%7, %15 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%17 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
          %19 = linalg.fill ins(%cst_3 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%18 : tensor<?x?x?xf16>) outs(%19 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %31 = arith.maximumf %in, %out : f16
            linalg.yield %31 : f16
          } -> tensor<?x?xf16>
          %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %20 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_7: f16, %out: f16):
            %31 = arith.truncf %cst_0 : f32 to f16
            %32 = arith.mulf %in_7, %31 : f16
            %33 = arith.cmpf ogt, %in, %32 : f16
            %34 = arith.select %33, %in, %32 : f16
            linalg.yield %34 : f16
          } -> tensor<?x?xf16>
          %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %21 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%16 : tensor<?x?x?xf16>) {
          ^bb0(%in: f16, %in_7: f16, %out: f16):
            %31 = arith.truncf %cst_0 : f32 to f16
            %32 = arith.mulf %in, %31 : f16
            %33 = arith.subf %32, %in_7 : f16
            %34 = math.powf %cst, %33 : f16
            linalg.yield %34 : f16
          } -> tensor<?x?x?xf16>
          %23 = linalg.fill ins(%cst_1 : f16) outs(%3 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%22 : tensor<?x?x?xf16>) outs(%23 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %31 = arith.addf %in, %out : f16
            linalg.yield %31 : f16
          } -> tensor<?x?xf16>
          %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %21 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_7: f16, %out: f16):
            %31 = arith.subf %in, %in_7 : f16
            %32 = math.powf %cst, %31 : f16
            linalg.yield %32 : f16
          } -> tensor<?x?xf16>
          %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %25, %24 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%3 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
            %31 = arith.mulf %in, %in_7 : f16
            %32 = arith.addf %31, %in_8 : f16
            linalg.yield %32 : f16
          } -> tensor<?x?xf16>
          %subview_6 = memref.subview %arg1[%5, %14, 0] [%0, %2, 128] [1, 1, 1] : memref<1x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
          %27 = bufferization.to_tensor %subview_6 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf16>
          %28 = linalg.fill ins(%cst_1 : f16) outs(%4 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
          %29 = linalg.batch_matmul ins(%22, %27 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%28 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
          %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %arg9, %25 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%4 : tensor<?x?x128xf16>) {
          ^bb0(%in: f16, %in_7: f16, %in_8: f16, %out: f16):
            %31 = arith.mulf %in_7, %in_8 : f16
            %32 = arith.addf %in, %31 : f16
            linalg.yield %32 : f16
          } -> tensor<?x?x128xf16>
          affine.yield %21, %26, %30 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
        }
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11#2, %11#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%4 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_5: f16, %out: f16):
          %14 = arith.divf %in, %in_5 : f16
          linalg.yield %14 : f16
        } -> tensor<?x?x128xf16>
        %subview_4 = memref.subview %arg3[%5, %6, 0] [%0, %1, 128] [1, 1, 1] : memref<1x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %13 = bufferization.to_buffer %12 : tensor<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        memref.copy %13, %subview_4 : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      }
      return
    }
  }
}
