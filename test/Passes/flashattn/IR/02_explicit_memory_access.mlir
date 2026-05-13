module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @attention(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c32 = arith.constant 32 : index
    %c4096 = arith.constant 4096 : index
    %0 = loom.sym @tile_b {upper_bound = 32 : index} : index
    %1 = loom.sym @tile_m {upper_bound = 4096 : index} : index
    %2 = loom.sym @tile_n {upper_bound = 4096 : index} : index
    %3 = arith.ceildivui %c32, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = arith.muli %arg4, %0 : index
      %6 = arith.muli %arg5, %1 : index
      %7 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf16>
      %8 = loom.semaphore_take %7 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %9 = loom.init_tensor %8[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %10 = loom.semaphore_take %7 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %11 = loom.init_tensor %10[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %12 = loom.semaphore_take %7 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %13 = loom.semaphore_take %7 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %14 = loom.init_tensor %13[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %15 = loom.subview %arg2[%5, %6, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      loom.copy %15, %12 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
      %16 = loom.bufferize_to_tensor %12[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %17 = loom.sync ins(%16 : tensor<?x?x128xf16>) outs(%14 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      loom.semaphore_give %12 : memref<?x?x128xf16>
      %18 = arith.ceildivui %c4096, %2 : index
      %19 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf16>
      %20 = loom.semaphore_take %19 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %21 = loom.init_tensor %20[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %22 = linalg.fill ins(%cst : f16) outs(%21 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %23 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %24 = loom.semaphore_take %23 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %25 = loom.init_tensor %24[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %26 = linalg.fill ins(%cst_0 : f16) outs(%25 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %27 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %28 = loom.semaphore_take %27 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %29 = loom.init_tensor %28[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %30 = linalg.fill ins(%cst_1 : f16) outs(%29 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %31 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf16>
      %32 = loom.semaphore_take %31 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %33 = loom.init_tensor %32[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %34 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %35 = loom.semaphore_take %34 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %36 = loom.init_tensor %35[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %37 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %38 = loom.semaphore_take %37 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %39 = loom.init_tensor %38[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %40 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %41 = loom.semaphore_take %40 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %42 = loom.init_tensor %41[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %43 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf16>
      %44 = loom.semaphore_take %43 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %45 = loom.init_tensor %44[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
      %46 = loom.semaphore_take %43 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %47 = loom.alloc [%0, %1, %2] on @L1 : memref<?x?x?xf16>
      %48 = loom.semaphore_take %47 : memref<?x?x?xf16> -> memref<?x?x?xf16>
      %49 = loom.init_tensor %48[%0, %1, %2] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
      %50 = loom.alloc [%0, %1, 32] on @L1 : memref<?x?x32xf16>
      %51 = loom.semaphore_take %50 : memref<?x?x32xf16> -> memref<?x?x32xf16>
      %52 = loom.init_tensor %51[%0, %1, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
      %53 = loom.semaphore_take %50 : memref<?x?x32xf16> -> memref<?x?x32xf16>
      %54 = loom.init_tensor %53[%0, %1, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
      %55 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf16>
      %56 = loom.semaphore_take %55 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %57 = loom.init_tensor %56[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %58 = loom.semaphore_take %55 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %59:3 = scf.for %arg6 = %c0 to %18 step %c1 iter_args(%arg7 = %30, %arg8 = %26, %arg9 = %22) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
        %65 = arith.muli %arg6, %2 : index
        %66 = loom.subview %arg0[%5, 0, %65] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        loom.copy %66, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
        %67 = loom.bufferize_to_tensor %46[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %68 = loom.sync ins(%67 : tensor<?x128x?xf16>) outs(%45 : tensor<?x128x?xf16>) -> tensor<?x128x?xf16>
        loom.semaphore_give %46 : memref<?x128x?xf16>
        %69 = linalg.fill ins(%cst : f16) outs(%49 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %70 = linalg.batch_matmul ins(%17, %68 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%69 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        loom.semaphore_give %44 : memref<?x128x?xf16>
        %71 = linalg.fill ins(%cst_1 : f16) outs(%36 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%70 : tensor<?x?x?xf16>) outs(%71 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %87 = arith.maximumf %in, %out : f16
          linalg.yield %87 : f16
        } -> tensor<?x?x1xf16>
        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %72 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%36 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %87 = arith.mulf %in_3, %cst_2 : f16
          %88 = arith.cmpf ogt, %in, %87 : f16
          %89 = arith.select %88, %in, %87 : f16
          linalg.yield %89 : f16
        } -> tensor<?x?x1xf16>
        %74 = loom.broadcast ins(%73 : tensor<?x?x1xf16>) outs(%54 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x?xf16>
        %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70, %74 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%49 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %87 = arith.mulf %in, %cst_2 : f16
          %88 = arith.subf %87, %in_3 : f16
          %89 = math.exp %88 : f16
          linalg.yield %89 : f16
        } -> tensor<?x?x?xf16>
        loom.semaphore_give %53 : memref<?x?x32xf16>
        %76 = linalg.fill ins(%cst : f16) outs(%39 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%75 : tensor<?x?x?xf16>) outs(%76 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %87 = arith.addf %in, %out : f16
          linalg.yield %87 : f16
        } -> tensor<?x?x1xf16>
        %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %73 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%42 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %87 = arith.subf %in, %in_3 : f16
          %88 = math.exp %87 : f16
          linalg.yield %88 : f16
        } -> tensor<?x?x1xf16>
        %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %78, %77 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg8 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %87 = arith.mulf %in, %in_3 : f16
          %88 = arith.addf %87, %in_4 : f16
          linalg.yield %88 : f16
        } -> tensor<?x?x1xf16>
        loom.semaphore_give %38 : memref<?x?x1xf16>
        %80 = loom.subview %arg1[%5, %65, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.copy %80, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %81 = loom.bufferize_to_tensor %58[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %82 = loom.sync ins(%81 : tensor<?x?x128xf16>) outs(%57 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        loom.semaphore_give %58 : memref<?x?x128xf16>
        %83 = linalg.fill ins(%cst : f16) outs(%33 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %84 = linalg.batch_matmul ins(%75, %82 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%83 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        loom.semaphore_give %56 : memref<?x?x128xf16>
        loom.semaphore_give %48 : memref<?x?x?xf16>
        %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%84, %arg9, %78 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg9 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %87 = arith.mulf %in_3, %in_4 : f16
          %88 = arith.addf %in, %87 : f16
          linalg.yield %88 : f16
        } -> tensor<?x?x128xf16>
        loom.semaphore_give %41 : memref<?x?x1xf16>
        loom.semaphore_give %32 : memref<?x?x128xf16>
        %86 = linalg.copy ins(%73 : tensor<?x?x1xf16>) outs(%arg7 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        loom.semaphore_give %35 : memref<?x?x1xf16>
        scf.yield %86, %79, %85 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
      }
      loom.semaphore_give %28 : memref<?x?x1xf16>
      loom.semaphore_give %13 : memref<?x?x128xf16>
      %60 = loom.broadcast ins(%59#1 : tensor<?x?x1xf16>) outs(%52 : tensor<?x?x32xf16>) dim(2) -> tensor<?x?x128xf16>
      loom.semaphore_give %24 : memref<?x?x1xf16>
      %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59#2, %60 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%11 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %65 = arith.divf %in, %in_3 : f16
        linalg.yield %65 : f16
      } -> tensor<?x?x128xf16>
      loom.semaphore_give %51 : memref<?x?x32xf16>
      loom.semaphore_give %20 : memref<?x?x128xf16>
      %62 = loom.sync ins(%61 : tensor<?x?x128xf16>) outs(%9 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      loom.semaphore_give %10 : memref<?x?x128xf16>
      %63 = loom.subview %arg3[%5, %6, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %64 = loom.bufferize_to_memref %62 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
      loom.copy %64, %63 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      loom.semaphore_give %8 : memref<?x?x128xf16>
    }
    return
  }
}
