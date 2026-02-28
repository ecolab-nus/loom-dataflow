module {
  module {
    func.func @attention(%arg0: memref<32x128x4096xf32>, %arg1: memref<32x4096x128xf32>, %arg2: memref<32x4096x128xf32>, %arg3: memref<32x4096x128xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.12751743074602467 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %0 = loom.get_symbolic_block_size @constraints::@BB : index
      %1 = loom.get_symbolic_block_size @constraints::@BM : index
      %2 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4, %arg5) = (0, 0) to (32 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = tensor.empty(%0, %1) : tensor<?x?xf32>
        %4 = tensor.empty(%0, %1) : tensor<?x?x128xf32>
        %5 = arith.muli %arg4, %0 : index
        %6 = arith.muli %arg5, %1 : index
        %subview = memref.subview %arg2[%5, %6, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
        %7 = bufferization.to_tensor %subview : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf32>
        %8 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
        %9 = linalg.fill ins(%cst_2 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %10 = linalg.fill ins(%cst_3 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %11:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %10, %arg8 = %9, %arg9 = %8) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
          %14 = arith.muli %arg6, %2 : index
          %subview_5 = memref.subview %arg0[%5, 0, %14] [%0, 128, %2] [1, 1, 1] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
          %15 = bufferization.to_tensor %subview_5 : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>> to tensor<?x128x?xf32>
          %16 = tensor.empty(%0, %1, %2) : tensor<?x?x?xf32>
          %17 = linalg.fill ins(%cst_1 : f32) outs(%16 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %18 = linalg.batch_matmul ins(%7, %15 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%17 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %19 = linalg.fill ins(%cst_3 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %20 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%18 : tensor<?x?x?xf32>) outs(%19 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %31 = arith.maximumf %in, %out : f32
            linalg.yield %31 : f32
          } -> tensor<?x?xf32>
          %21 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %20 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %31 = arith.truncf %cst_0 : f64 to f32
            %32 = arith.mulf %in_7, %31 : f32
            %33 = arith.cmpf ogt, %in, %32 : f32
            %34 = arith.select %33, %in, %32 : f32
            linalg.yield %34 : f32
          } -> tensor<?x?xf32>
          %22 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%18, %21 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %31 = arith.truncf %cst_0 : f64 to f32
            %32 = arith.mulf %in, %31 : f32
            %33 = arith.subf %32, %in_7 : f32
            %34 = math.powf %cst, %33 : f32
            linalg.yield %34 : f32
          } -> tensor<?x?x?xf32>
          %23 = linalg.fill ins(%cst_1 : f32) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %24 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%22 : tensor<?x?x?xf32>) outs(%23 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %31 = arith.addf %in, %out : f32
            linalg.yield %31 : f32
          } -> tensor<?x?xf32>
          %25 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %21 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_7: f32, %out: f32):
            %31 = arith.subf %in, %in_7 : f32
            %32 = math.powf %cst, %31 : f32
            linalg.yield %32 : f32
          } -> tensor<?x?xf32>
          %26 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %25, %24 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %31 = arith.mulf %in, %in_7 : f32
            %32 = arith.addf %31, %in_8 : f32
            linalg.yield %32 : f32
          } -> tensor<?x?xf32>
          %subview_6 = memref.subview %arg1[%5, %14, 0] [%0, %2, 128] [1, 1, 1] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
          %27 = bufferization.to_tensor %subview_6 : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>> to tensor<?x?x128xf32>
          %28 = linalg.fill ins(%cst_1 : f32) outs(%4 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
          %29 = linalg.batch_matmul ins(%22, %27 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%28 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
          %30 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %arg9, %25 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?x128xf32>) {
          ^bb0(%in: f32, %in_7: f32, %in_8: f32, %out: f32):
            %31 = arith.mulf %in_7, %in_8 : f32
            %32 = arith.addf %in, %31 : f32
            linalg.yield %32 : f32
          } -> tensor<?x?x128xf32>
          affine.yield %21, %26, %30 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
        }
        %12 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11#2, %11#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?x128xf32>) {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %14 = arith.divf %in, %in_5 : f32
          linalg.yield %14 : f32
        } -> tensor<?x?x128xf32>
        %subview_4 = memref.subview %arg3[%5, %6, 0] [%0, %1, 128] [1, 1, 1] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
        %13 = bufferization.to_buffer %12 : tensor<?x?x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
        memref.copy %13, %subview_4 : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
      }
      return
    }
  }
}
