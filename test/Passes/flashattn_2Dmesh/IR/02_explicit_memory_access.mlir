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
        %5 = loom.semaphore %4 : memref<?x?x?xf32> -> memref<?x?x?xf32>
        %6 = loom.init_tensor %5[%0, %1, %2] : memref<?x?x?xf32> -> tensor<?x?x?xf32>
        %7 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf32>
        %8 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %9 = loom.semaphore %8 : memref<?x?xf32> -> memref<?x?xf32>
        %10 = loom.init_tensor %9[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %11 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %12 = loom.semaphore %11 : memref<?x?xf32> -> memref<?x?xf32>
        %13 = loom.init_tensor %12[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %14 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %15 = loom.semaphore %14 : memref<?x?xf32> -> memref<?x?xf32>
        %16 = loom.init_tensor %15[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %17 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %18 = loom.semaphore %17 : memref<?x?xf32> -> memref<?x?xf32>
        %19 = loom.init_tensor %18[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %20 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
        %21 = loom.semaphore %20 : memref<?x?xf32> -> memref<?x?xf32>
        %22 = loom.init_tensor %21[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
        %23 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf32>
        %24 = loom.semaphore %23 : memref<?x?x128xf32> -> memref<?x?x128xf32>
        %25 = loom.init_tensor %24[%0, %1, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
        %26 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf32>
        %27 = loom.semaphore %26 : memref<?x?x128xf32> -> memref<?x?x128xf32>
        %28 = loom.init_tensor %27[%0, %1, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
        %29 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf32>
        %30 = loom.semaphore %29 : memref<?x?x128xf32> -> memref<?x?x128xf32>
        %31 = loom.init_tensor %30[%0, %1, 128] : memref<?x?x128xf32> -> tensor<?x?x128xf32>
        %32 = arith.muli %arg4, %0 : index
        %33 = arith.muli %arg5, %1 : index
        %34 = loom.subview %arg2[%32, %33, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
        %35 = loom.semaphore %23 : memref<?x?x128xf32> -> memref<?x?x128xf32>
        %36 = loom.copy_to_tensor %34, %35, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
        %37 = linalg.fill ins(%cst_1 : f32) outs(%28 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
        %38 = linalg.fill ins(%cst_2 : f32) outs(%10 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %39 = linalg.fill ins(%cst_3 : f32) outs(%13 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %40:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %39, %arg8 = %38, %arg9 = %37) -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>) {
          %43 = arith.muli %arg6, %2 : index
          %44 = loom.subview %arg0[%32, 0, %43] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf32> to memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>
          %45 = loom.semaphore %7 : memref<?x128x?xf32> -> memref<?x128x?xf32>
          %46 = loom.copy_to_tensor %44, %45, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf32, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf32> -> tensor<?x128x?xf32>
          %47 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %48 = linalg.batch_matmul ins(%36, %46 : tensor<?x?x128xf32>, tensor<?x128x?xf32>) outs(%47 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
          %49 = linalg.fill ins(%cst_3 : f32) outs(%16 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %50 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : tensor<?x?x?xf32>) outs(%49 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %64 = arith.maximumf %in, %out : f32
            linalg.yield %64 : f32
          } -> tensor<?x?xf32>
          %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %50 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%16 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %64 = arith.truncf %cst_0 : f64 to f32
            %65 = arith.mulf %in_4, %64 : f32
            %66 = arith.cmpf ogt, %in, %65 : f32
            %67 = arith.select %66, %in, %65 : f32
            linalg.yield %67 : f32
          } -> tensor<?x?xf32>
          %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %51 : tensor<?x?x?xf32>, tensor<?x?xf32>) outs(%6 : tensor<?x?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %64 = arith.truncf %cst_0 : f64 to f32
            %65 = arith.mulf %in, %64 : f32
            %66 = arith.subf %65, %in_4 : f32
            %67 = math.powf %cst, %66 : f32
            linalg.yield %67 : f32
          } -> tensor<?x?x?xf32>
          %53 = linalg.fill ins(%cst_1 : f32) outs(%19 : tensor<?x?xf32>) -> tensor<?x?xf32>
          %54 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : tensor<?x?x?xf32>) outs(%53 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %out: f32):
            %64 = arith.addf %in, %out : f32
            linalg.yield %64 : f32
          } -> tensor<?x?xf32>
          %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %51 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%22 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %out: f32):
            %64 = arith.subf %in, %in_4 : f32
            %65 = math.powf %cst, %64 : f32
            linalg.yield %65 : f32
          } -> tensor<?x?xf32>
          %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %55, %54 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg8 : tensor<?x?xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %64 = arith.mulf %in, %in_4 : f32
            %65 = arith.addf %64, %in_5 : f32
            linalg.yield %65 : f32
          } -> tensor<?x?xf32>
          %57 = loom.subview %arg1[%32, %43, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
          %58 = loom.semaphore %3 : memref<?x?x128xf32> -> memref<?x?x128xf32>
          %59 = loom.copy_to_tensor %57, %58, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf32> -> tensor<?x?x128xf32>
          %60 = linalg.fill ins(%cst_1 : f32) outs(%31 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
          %61 = linalg.batch_matmul ins(%52, %59 : tensor<?x?x?xf32>, tensor<?x?x128xf32>) outs(%60 : tensor<?x?x128xf32>) -> tensor<?x?x128xf32>
          %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %arg9, %55 : tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%arg9 : tensor<?x?x128xf32>) {
          ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32):
            %64 = arith.mulf %in_4, %in_5 : f32
            %65 = arith.addf %in, %64 : f32
            linalg.yield %65 : f32
          } -> tensor<?x?x128xf32>
          %63 = linalg.copy ins(%51 : tensor<?x?xf32>) outs(%arg7 : tensor<?x?xf32>) -> tensor<?x?xf32>
          affine.yield %63, %56, %62 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?x128xf32>
        }
        %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%40#2, %40#1 : tensor<?x?x128xf32>, tensor<?x?xf32>) outs(%25 : tensor<?x?x128xf32>) {
        ^bb0(%in: f32, %in_4: f32, %out: f32):
          %43 = arith.divf %in, %in_4 : f32
          linalg.yield %43 : f32
        } -> tensor<?x?x128xf32>
        %42 = loom.subview %arg3[%32, %33, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf32> to memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
        loom.copy_from_tensor %41, %42 : tensor<?x?x128xf32>, memref<?x?x128xf32, strided<[524288, 128, 1], offset: ?>>
      }
      return
    }
  }
}
