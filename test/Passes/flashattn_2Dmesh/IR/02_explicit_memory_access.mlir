module {
  module {
    loom.constraint_space @constraints {
      %0 = loom.symbolic_var "BB" : index
      %1 = loom.symbolic_var "BM" : index
      %2 = loom.symbolic_var "BN" : index
      loom.range %0[0, 2]
      loom.range %1[0, 4096]
      loom.range %2[0, 4096]
    }
    func.func @attention(%arg0: memref<2x4096x4096xf32>, %arg1: memref<2x4096x4096xf32>, %arg2: memref<2x4096x4096xf32>, %arg3: memref<2x4096x4096xf32>) {
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 0.022542110000000001 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %cst_2 = arith.constant 1.000000e+00 : f32
      %cst_3 = arith.constant 0xFF800000 : f32
      %0 = loom.get_symbolic_block_size @constraints::@BB : index
      %1 = loom.get_symbolic_block_size @constraints::@BM : index
      %2 = loom.get_symbolic_block_size @constraints::@BN : index
      affine.parallel (%arg4, %arg5) = (0, 0) to (2 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = loom.alloc [%0, %2, 4096] on @L1 : memref<?x?x4096xf32>
        %4 = loom.alloc [%0, %1, %2] on @L1 : memref<?x?x?xf32>
        %5 = loom.init_tensor %4[%0, %1, %2] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
        %6 = loom.alloc [%0, 4096, %2] on @L1 : memref<?x4096x?xf32>
        %7 = loom.alloc [%0, %1, 4096] on @L1 : memref<?x?x4096xf32>
        %8 = loom.init_tensor %7[%0, %1, 4096] : memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
        %9 = loom.alloc [%0, %1, 4096] on @L1 : memref<?x?x4096xf32>
        %10 = loom.alloc [%0, %1, 4096] on @L1 : memref<?x?x4096xf32>
        %11 = loom.init_tensor %10[%0, %1, 4096] : memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
        %12 = loom.alloc [%0, %1, 4096] on @L1 : memref<?x?x4096xf32>
        %13 = loom.init_tensor %12[%0, %1, 4096] : memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
        %14 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %15 = loom.init_tensor %14[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %16 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %17 = loom.init_tensor %16[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %18 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %19 = loom.init_tensor %18[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %20 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %21 = loom.init_tensor %20[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %22 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %23 = loom.init_tensor %22[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %24 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %25 = loom.init_tensor %24[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %26 = linalg.fill ins(%cst_3 : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %27 = linalg.fill ins(%cst_3 : f32) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %28 = linalg.fill ins(%cst_2 : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %29 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
        %30 = linalg.fill ins(%cst_1 : f32) outs(%11 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
        %31 = arith.muli %arg4, %0 : index
        %32 = arith.muli %arg5, %1 : index
        %33 = loom.subview %arg2[%31, %32, 0] [%0, %1, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %34 = loom.copy_to_tensor %33, %9, interconnect : [], broadcast : [1, 1] : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
        %35:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %27, %arg8 = %28, %arg9 = %30) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>) {
          %38 = arith.muli %arg6, %2 : index
          %39 = loom.subview %arg0[%31, 0, %38] [%0, 4096, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>>
          %40 = loom.copy_to_tensor %39, %6, interconnect : [], broadcast : [1, 1] : memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x4096x?xf32> -> tensor<?x4096x?xf32>
          %41 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %42 = linalg.batch_matmul ins(%34, %40 : tensor<?x?x4096xf32>, tensor<?x4096x?xf32>) outs(%41 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %43 = linalg.copy ins(%26 : tensor<?x?xf32>) outs(%21 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%42 : tensor<?x?x?xf32>) outs(%43 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %57 = arith.maximumf %in, %out : f32
            linalg.yield %57 : f32
          } -> tensor<?x?xf32>
          %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %44 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%21 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %57 = arith.truncf %cst_0 : f64 to f32
            %58 = arith.mulf %in_4, %57 : f32
            %59 = arith.cmpf ogt, %in, %58 : f32
            %60 = arith.select %59, %in, %58 : f32
            linalg.yield %60 : f32
          } -> tensor<?x?xf32>
          %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42, %45 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %57 = arith.truncf %cst_0 : f64 to f32
            %58 = arith.mulf %in, %57 : f32
            %59 = arith.subf %58, %in_4 : f32
            %60 = math.powf %cst, %59 : f32
            linalg.yield %60 : f32
          } -> tensor<?x?x?xf32>
          %47 = linalg.fill ins(%cst_1 : f32) outs(%23 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : tensor<?x?x?xf32>) outs(%47 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %57 = arith.addf %in, %out : f32
            linalg.yield %57 : f32
          } -> tensor<?x?xf32>
          %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %45 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%25 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %57 = arith.subf %in, %in_4 : f32
            %58 = math.powf %cst, %57 : f32
            linalg.yield %58 : f32
          } -> tensor<?x?xf32>
          %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %49, %48 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %57 = arith.mulf %in, %in_4 : f32
            %58 = arith.addf %57, %in_5 : f32
            linalg.yield %58 : f32
          } -> tensor<?x?xf32>
          %51 = loom.subview %arg1[%31, %38, 0] [%0, %2, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          %52 = loom.copy_to_tensor %51, %3, interconnect : [], broadcast : [1, 1] : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
          %53 = linalg.copy ins(%29 : tensor<?x?x4096xf32>) outs(%13 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
          %54 = linalg.batch_matmul ins(%46, %52 : tensor<?x?x?xf32>, tensor<?x?x4096xf32>) outs(%53 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
          %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%54, %arg9, %49 : tensor<?x?x4096xf32>, tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?x4096xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %57 = arith.mulf %in_4, %in_5 : f32
            %58 = arith.addf %in, %57 : f32
            linalg.yield %58 : f32
          } -> tensor<?x?x4096xf32>
          %56 = linalg.copy ins(%45 : tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
          affine.yield %56, %50, %55 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>
        }
        %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35#2, %35#1 : tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?x4096xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %38 = arith.divf %in, %in_4 : f32
          linalg.yield %38 : f32
        } -> tensor<?x?x4096xf32>
        %37 = loom.subview %arg3[%31, %32, 0] [%0, %1, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy_from_tensor %36, %37 : tensor<?x?x4096xf32>, memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
      }
      return
    }
  }
}
