module {
  module {
    loom.constraint_space @constraints {
      %0 = loom.symbolic_var "BB" : index
      %1 = loom.symbolic_var "BM" : index
      %2 = loom.symbolic_var "BN" : index
      loom.range %0[0, 32]
      loom.range %1[0, 4096]
      loom.range %2[0, 4096]
    }
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
        %3 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf32>
        %4 = loom.alloc [%0, %1, %2] on @L1 : memref<?x?x?xf32>
        %5 = loom.init_tensor %4[%0, %1, %2] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
        %6 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf32>
        %7 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %8 = loom.init_tensor %7[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %9 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %10 = loom.init_tensor %9[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %11 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %12 = loom.init_tensor %11[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %13 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %14 = loom.init_tensor %13[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %15 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %16 = loom.init_tensor %15[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %17 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf32>
        %18 = loom.init_tensor %17[%0, %1, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
        %19 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf32>
        %20 = loom.init_tensor %19[%0, %1, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
        %21 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf32>
        %22 = loom.init_tensor %21[%0, %1, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
        %23 = arith.muli %arg4, %0 : index
        %24 = arith.muli %arg5, %1 : index
        %25 = loom.subview %arg2[%23, %24, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
        %26 = loom.copy_to_tensor %25, %17, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
        %27 = linalg.fill ins(%cst_1 : f32) outs(%20 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
        %28 = linalg.fill ins(%cst_2 : f32) outs(%8 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %29 = linalg.fill ins(%cst_3 : f32) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %30:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %29, %arg8 = %28, %arg9 = %27) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
          %33 = arith.muli %arg6, %2 : index
          %34 = loom.subview %arg0[%23, 0, %33] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
          %35 = loom.copy_to_tensor %34, %6, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
          %36 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %37 = linalg.batch_matmul ins(%26, %35 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%36 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %38 = linalg.fill ins(%cst_3 : f32) outs(%12 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %39 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%37 : tensor<?x?x?xf32>) outs(%38 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %52 = arith.maximumf %in, %out : f32
            linalg.yield %52 : f32
          } -> tensor<?x?xf32>
          %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %39 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%12 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %52 = arith.truncf %cst_0 : f64 to f32
            %53 = arith.mulf %in_4, %52 : f32
            %54 = arith.cmpf ogt, %in, %53 : f32
            %55 = arith.select %54, %in, %53 : f32
            linalg.yield %55 : f32
          } -> tensor<?x?xf32>
          %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %40 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %52 = arith.truncf %cst_0 : f64 to f32
            %53 = arith.mulf %in, %52 : f32
            %54 = arith.subf %53, %in_4 : f32
            %55 = math.powf %cst, %54 : f32
            linalg.yield %55 : f32
          } -> tensor<?x?x?xf32>
          %42 = linalg.fill ins(%cst_1 : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%41 : tensor<?x?x?xf32>) outs(%42 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %52 = arith.addf %in, %out : f32
            linalg.yield %52 : f32
          } -> tensor<?x?xf32>
          %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %40 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %52 = arith.subf %in, %in_4 : f32
            %53 = math.powf %cst, %52 : f32
            linalg.yield %53 : f32
          } -> tensor<?x?xf32>
          %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %44, %43 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg8 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %52 = arith.mulf %in, %in_4 : f32
            %53 = arith.addf %52, %in_5 : f32
            linalg.yield %53 : f32
          } -> tensor<?x?xf32>
          %46 = loom.subview %arg1[%23, %33, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
          %47 = loom.copy_to_tensor %46, %3, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
          %48 = linalg.fill ins(%cst_1 : f32) outs(%22 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
          %49 = linalg.batch_matmul ins(%41, %47 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%48 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
          %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %arg9, %44 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg9 : tensor<?x?x128xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %52 = arith.mulf %in_4, %in_5 : f32
            %53 = arith.addf %in, %52 : f32
            linalg.yield %53 : f32
          } -> tensor<?x?x128xf32>
          %51 = linalg.copy ins(%40 : tensor<?x?xf32>) outs(%arg7 : tensor<?x?xf32>) -> tensor<?x?xf32>
          affine.yield %51, %45, %50 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
        }
        %31 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%30#2, %30#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?x128xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %33 = arith.divf %in, %in_4 : f32
          linalg.yield %33 : f32
        } -> tensor<?x?x128xf32>
        %32 = loom.subview %arg3[%23, %24, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
        loom.copy_from_tensor %31, %32 : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
      }
      return
    }
  }
}
