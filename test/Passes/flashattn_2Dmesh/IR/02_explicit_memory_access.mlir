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
        %13 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %14 = loom.init_tensor %13[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %15 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %16 = loom.init_tensor %15[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %17 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %18 = loom.init_tensor %17[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %19 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %20 = loom.init_tensor %19[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %21 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %22 = loom.init_tensor %21[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %23 = linalg.fill ins(%cst_3 : f32) outs(%14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %24 = linalg.fill ins(%cst_3 : f32) outs(%18 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %25 = linalg.fill ins(%cst_2 : f32) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %26 = linalg.fill ins(%cst_1 : f32) outs(%8 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
        %27 = linalg.fill ins(%cst_1 : f32) outs(%11 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
        %28 = arith.muli %arg4, %0 : index
        %29 = arith.muli %arg5, %1 : index
        %30 = loom.subview %arg2[%28, %29, 0] [%0, %1, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        %31 = loom.copy_to_tensor %30, %9, interconnect : [], broadcast : [1, 1] : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
        %32:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %24, %arg8 = %25, %arg9 = %27) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>) {
          %39 = arith.muli %arg6, %2 : index
          %40 = loom.subview %arg0[%28, 0, %39] [%0, 4096, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>>
          %41 = loom.copy_to_tensor %40, %6, interconnect : [], broadcast : [1, 1] : memref<?x4096x?xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x4096x?xf32> -> tensor<?x4096x?xf32>
          %42 = linalg.fill ins(%cst_1 : f32) outs(%5 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %43 = linalg.batch_matmul ins(%31, %41 : tensor<?x?x4096xf32>, tensor<?x4096x?xf32>) outs(%42 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %44 = loom.pb_anchor %43, %4 : tensor<?x?x?xf32>, memref<?x?x?xf32> -> tensor<?x?x?xf32>
          %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%44 : tensor<?x?x?xf32>) outs(%23 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %61 = arith.maximumf %in, %out : f32
            linalg.yield %61 : f32
          } -> tensor<?x?xf32>
          %46 = loom.pb_anchor %45, %19 : tensor<?x?xf32>, memref<?x?xf32> -> tensor<?x?xf32>
          %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %46 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%18 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %61 = arith.truncf %cst_0 : f64 to f32
            %62 = arith.mulf %in_4, %61 : f32
            %63 = arith.cmpf ogt, %in, %62 : f32
            %64 = arith.select %63, %in, %62 : f32
            linalg.yield %64 : f32
          } -> tensor<?x?xf32>
          %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%44, %47 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%5 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %61 = arith.truncf %cst_0 : f64 to f32
            %62 = arith.mulf %in, %61 : f32
            %63 = arith.subf %62, %in_4 : f32
            %64 = math.powf %cst, %63 : f32
            linalg.yield %64 : f32
          } -> tensor<?x?x?xf32>
          %49 = loom.pb_anchor %48, %4 : tensor<?x?x?xf32>, memref<?x?x?xf32> -> tensor<?x?x?xf32>
          %50 = linalg.fill ins(%cst_1 : f32) outs(%20 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : tensor<?x?x?xf32>) outs(%50 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %61 = arith.addf %in, %out : f32
            linalg.yield %61 : f32
          } -> tensor<?x?xf32>
          %52 = loom.pb_anchor %51, %19 : tensor<?x?xf32>, memref<?x?xf32> -> tensor<?x?xf32>
          %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %47 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%22 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %61 = arith.subf %in, %in_4 : f32
            %62 = math.powf %cst, %61 : f32
            linalg.yield %62 : f32
          } -> tensor<?x?xf32>
          %54 = loom.pb_anchor %53, %21 : tensor<?x?xf32>, memref<?x?xf32> -> tensor<?x?xf32>
          %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %54, %52 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %61 = arith.mulf %in, %in_4 : f32
            %62 = arith.addf %61, %in_5 : f32
            linalg.yield %62 : f32
          } -> tensor<?x?xf32>
          %56 = loom.subview %arg1[%28, %39, 0] [%0, %2, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
          %57 = loom.copy_to_tensor %56, %3, interconnect : [], broadcast : [1, 1] : memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
          %58 = linalg.batch_matmul ins(%49, %57 : tensor<?x?x?xf32>, tensor<?x?x4096xf32>) outs(%26 : tensor<?x?x4096xf32>) -> tensor<?x?x4096xf32>
          %59 = loom.pb_anchor %58, %12 : tensor<?x?x4096xf32>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
          %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %arg9, %54 : tensor<?x?x4096xf32>, tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?x4096xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %61 = arith.mulf %in_4, %in_5 : f32
            %62 = arith.addf %in, %61 : f32
            linalg.yield %62 : f32
          } -> tensor<?x?x4096xf32>
          affine.yield %47, %55, %60 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x4096xf32>
        }
        %33 = loom.pb_anchor %32#0, %17 : tensor<?x?xf32>, memref<?x?xf32> -> tensor<?x?xf32>
        %34 = loom.pb_anchor %32#1, %15 : tensor<?x?xf32>, memref<?x?xf32> -> tensor<?x?xf32>
        %35 = loom.pb_anchor %32#2, %10 : tensor<?x?x4096xf32>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
        %36 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %34 : tensor<?x?x4096xf32>, tensor<?x?xf32>) outs(%8 : tensor<?x?x4096xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %39 = arith.divf %in, %in_4 : f32
          linalg.yield %39 : f32
        } -> tensor<?x?x4096xf32>
        %37 = loom.pb_anchor %36, %7 : tensor<?x?x4096xf32>, memref<?x?x4096xf32> -> tensor<?x?x4096xf32>
        %38 = loom.subview %arg3[%28, %29, 0] [%0, %1, 4096] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4096x4096xf32> to memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
        loom.copy_from_tensor %37, %38 : tensor<?x?x4096xf32>, memref<?x?x4096xf32, strided<[16777216, 4096, 1], offset: ?>>
      }
      return
    }
  }
}
