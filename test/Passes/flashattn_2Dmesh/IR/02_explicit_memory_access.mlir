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
        %24 = linalg.fill ins(%cst_3 : f32) outs(%15 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %25 = linalg.fill ins(%cst_3 : f32) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %26 = linalg.fill ins(%cst_2 : f32) outs(%17 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %27 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
        %28 = linalg.fill ins(%cst_1 : f32) outs(%11 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
        %29 = arith.muli %arg4, %0 : index
        %30 = arith.muli %arg5, %1 : index
        %31 = loom.subview %arg2[%29, %30, 0] [%0, %1, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %32 = loom.copy_to_tensor %31, %9, interconnect : [], broadcast : [1, 1] : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
        %33:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %25, %arg8 = %26, %arg9 = %28) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>) {
          %36 = arith.muli %arg6, %2 : index
          %37 = loom.subview %arg0[%29, 0, %36] [%0, 4096, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>>
          %38 = loom.copy_to_tensor %37, %6, interconnect : [], broadcast : [1, 1] : memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x4096x?xf32> -> tensor<?x4096x?xf32>
          %39 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %40 = linalg.batch_matmul ins(%32, %38 : tensor<?x?x4096xf32>, tensor<?x4096x?xf32>) outs(%39 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %41 = linalg.copy ins(%24 : tensor<?x?xf32>) outs(%21 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %42 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%40 : tensor<?x?x?xf32>) outs(%41 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %54 = arith.maximumf %in, %out : f32
            linalg.yield %54 : f32
          } -> tensor<?x?xf32>
          %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %42 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%19 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %54 = arith.truncf %cst_0 : f64 to f32
            %55 = arith.mulf %in_4, %54 : f32
            %56 = arith.cmpf ogt, %in, %55 : f32
            %57 = arith.select %56, %in, %55 : f32
            linalg.yield %57 : f32
          } -> tensor<?x?xf32>
          %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40, %43 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %54 = arith.truncf %cst_0 : f64 to f32
            %55 = arith.mulf %in, %54 : f32
            %56 = arith.subf %55, %in_4 : f32
            %57 = math.powf %cst, %56 : f32
            linalg.yield %57 : f32
          } -> tensor<?x?x?xf32>
          %45 = linalg.fill ins(%cst_1 : f32) outs(%21 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%44 : tensor<?x?x?xf32>) outs(%45 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %54 = arith.addf %in, %out : f32
            linalg.yield %54 : f32
          } -> tensor<?x?xf32>
          %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %43 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%23 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %54 = arith.subf %in, %in_4 : f32
            %55 = math.powf %cst, %54 : f32
            linalg.yield %55 : f32
          } -> tensor<?x?xf32>
          %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %47, %46 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%17 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %54 = arith.mulf %in, %in_4 : f32
            %55 = arith.addf %54, %in_5 : f32
            linalg.yield %55 : f32
          } -> tensor<?x?xf32>
          %49 = loom.subview %arg1[%29, %36, 0] [%0, %2, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          %50 = loom.copy_to_tensor %49, %3, interconnect : [], broadcast : [1, 1] : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
          %51 = linalg.copy ins(%27 : tensor<?x?x4096xf32>) outs(%13 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
          %52 = linalg.batch_matmul ins(%44, %50 : tensor<?x?x?xf32>, tensor<?x?x4096xf32>) outs(%51 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
          %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52, %arg9, %47 : tensor<?x?x4096xf32>, tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?x4096xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %54 = arith.mulf %in_4, %in_5 : f32
            %55 = arith.addf %in, %54 : f32
            linalg.yield %55 : f32
          } -> tensor<?x?x4096xf32>
          affine.yield %43, %48, %53 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>
        }
        %34 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33#2, %33#1 : tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?x4096xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %36 = arith.divf %in, %in_4 : f32
          linalg.yield %36 : f32
        } -> tensor<?x?x4096xf32>
        %35 = loom.subview %arg3[%29, %30, 0] [%0, %1, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy_from_tensor %34, %35 : tensor<?x?x4096xf32>, memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
      }
      return
    }
  }
}
