module {
  module {
    func.func @attention(%arg0: memref<1x128x4096xf16>, %arg1: memref<1x4096x128xf16>, %arg2: memref<1x4096x128xf16>, %arg3: memref<1x4096x128xf16>) {
      %cst = arith.constant 2.000000e+00 : f16
      %cst_0 = arith.constant 1.275630e-01 : f16
      %cst_1 = arith.constant 0.000000e+00 : f16
      %cst_2 = arith.constant 1.000000e+00 : f16
      %cst_3 = arith.constant 0xFC00 : f16
      %0 = loom.sym @block_size_0 : index
      %1 = loom.sym @block_size_1 : index
      %2 = loom.sym @block_size_2 : index
      affine.parallel (%arg4, %arg5) = (0, 0) to (1 ceildiv symbol(%0), 4096 ceildiv symbol(%1)) {
        %3 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf16>
        %4 = loom.semaphore_take %3 : memref<?x?x128xf16> -> memref<?x?x128xf16>
        %5 = loom.alloc [%0, %1, %2] on @L1 : memref<?x?x?xf16>
        %6 = loom.semaphore_take %5 : memref<?x?x?xf16> -> memref<?x?x?xf16>
        %7 = loom.init_tensor %6[%0, %1, %2] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
        %8 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf16>
        %9 = loom.semaphore_take %8 : memref<?x128x?xf16> -> memref<?x128x?xf16>
        %10 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
        %11 = loom.semaphore_take %10 : memref<?x?xf16> -> memref<?x?xf16>
        %12 = loom.init_tensor %11[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %13 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
        %14 = loom.semaphore_take %13 : memref<?x?xf16> -> memref<?x?xf16>
        %15 = loom.init_tensor %14[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %16 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
        %17 = loom.semaphore_take %16 : memref<?x?xf16> -> memref<?x?xf16>
        %18 = loom.init_tensor %17[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %19 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
        %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
        %21 = loom.init_tensor %20[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %22 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
        %23 = loom.semaphore_take %22 : memref<?x?xf16> -> memref<?x?xf16>
        %24 = loom.init_tensor %23[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %25 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf16>
        %26 = loom.semaphore_take %25 : memref<?x?x128xf16> -> memref<?x?x128xf16>
        %27 = loom.init_tensor %26[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %28 = loom.semaphore_take %25 : memref<?x?x128xf16> -> memref<?x?x128xf16>
        %29 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf16>
        %30 = loom.semaphore_take %29 : memref<?x?x128xf16> -> memref<?x?x128xf16>
        %31 = loom.init_tensor %30[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %32 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf16>
        %33 = loom.semaphore_take %32 : memref<?x?x128xf16> -> memref<?x?x128xf16>
        %34 = loom.init_tensor %33[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %35 = arith.muli %arg4, %0 : index
        %36 = arith.muli %arg5, %1 : index
        %37 = loom.subview %arg2[%35, %36, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        %38 = loom.copy_to_tensor %37, %28, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %39 = linalg.fill ins(%cst_1 : f16) outs(%31 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %40 = linalg.fill ins(%cst_2 : f16) outs(%12 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %41 = linalg.fill ins(%cst_3 : f16) outs(%15 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %42:3 = affine.for %arg6 = 0 to affine_map<()[s0] -> (4096 ceildiv s0)>()[%2] iter_args(%arg7 = %41, %arg8 = %40, %arg9 = %39) -> (tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>) {
          %45 = arith.muli %arg6, %2 : index
          %46 = loom.subview %arg0[%35, 0, %45] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
          %47 = loom.copy_to_tensor %46, %9, interconnect : [], broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>, memref<?x128x?xf16> -> tensor<?x128x?xf16>
          %48 = linalg.fill ins(%cst_1 : f16) outs(%7 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
          %49 = linalg.batch_matmul ins(%38, %47 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%48 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
          loom.semaphore_give %9 : memref<?x128x?xf16>
          %50 = linalg.fill ins(%cst_3 : f16) outs(%18 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %51 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%49 : tensor<?x?x?xf16>) outs(%50 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %64 = arith.maximumf %in, %out : f16
            linalg.yield %64 : f16
          } -> tensor<?x?xf16>
          %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %51 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%18 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %64 = arith.mulf %in_4, %cst_0 : f16
            %65 = arith.cmpf ogt, %in, %64 : f16
            %66 = arith.select %65, %in, %64 : f16
            linalg.yield %66 : f16
          } -> tensor<?x?xf16>
          %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %52 : tensor<?x?x?xf16>, tensor<?x?xf16>) outs(%7 : tensor<?x?x?xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %64 = arith.mulf %in, %cst_0 : f16
            %65 = arith.subf %64, %in_4 : f16
            %66 = math.powf %cst, %65 : f16
            linalg.yield %66 : f16
          } -> tensor<?x?x?xf16>
          %54 = linalg.fill ins(%cst_1 : f16) outs(%21 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %55 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%53 : tensor<?x?x?xf16>) outs(%54 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %out: f16):
            %64 = arith.addf %in, %out : f16
            linalg.yield %64 : f16
          } -> tensor<?x?xf16>
          %56 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %52 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%24 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %64 = arith.subf %in, %in_4 : f16
            %65 = math.powf %cst, %64 : f16
            linalg.yield %65 : f16
          } -> tensor<?x?xf16>
          %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %56, %55 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg8 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
            %64 = arith.mulf %in, %in_4 : f16
            %65 = arith.addf %64, %in_5 : f16
            linalg.yield %65 : f16
          } -> tensor<?x?xf16>
          loom.semaphore_give %20 : memref<?x?xf16>
          %58 = loom.subview %arg1[%35, %45, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
          %59 = loom.copy_to_tensor %58, %4, interconnect : [], broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>, memref<?x?x128xf16> -> tensor<?x?x128xf16>
          %60 = linalg.fill ins(%cst_1 : f16) outs(%34 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
          %61 = linalg.batch_matmul ins(%53, %59 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%60 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
          loom.semaphore_give %4 : memref<?x?x128xf16>
          loom.semaphore_give %6 : memref<?x?x?xf16>
          %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%61, %arg9, %56 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%arg9 : tensor<?x?x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
            %64 = arith.mulf %in_4, %in_5 : f16
            %65 = arith.addf %in, %64 : f16
            linalg.yield %65 : f16
          } -> tensor<?x?x128xf16>
          loom.semaphore_give %23 : memref<?x?xf16>
          loom.semaphore_give %33 : memref<?x?x128xf16>
          %63 = linalg.copy ins(%52 : tensor<?x?xf16>) outs(%arg7 : tensor<?x?xf16>) -> tensor<?x?xf16>
          loom.semaphore_give %17 : memref<?x?xf16>
          affine.yield %63, %57, %62 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?x128xf16>
        }
        loom.semaphore_give %14 : memref<?x?xf16>
        loom.semaphore_give %28 : memref<?x?x128xf16>
        %43 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%42#2, %42#1 : tensor<?x?x128xf16>, tensor<?x?xf16>) outs(%27 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %45 = arith.divf %in, %in_4 : f16
          linalg.yield %45 : f16
        } -> tensor<?x?x128xf16>
        loom.semaphore_give %11 : memref<?x?xf16>
        loom.semaphore_give %30 : memref<?x?x128xf16>
        %44 = loom.subview %arg3[%35, %36, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.copy_from_tensor %43, %44 : tensor<?x?x128xf16>, memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.semaphore_give %26 : memref<?x?x128xf16>
      }
      return
    }
  }
}
