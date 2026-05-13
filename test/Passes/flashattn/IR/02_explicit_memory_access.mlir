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
      %9 = loom.subview %arg2[%5, %6, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      loom.copy %9, %8 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
      %10 = loom.bufferize_to_tensor %8[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %11 = arith.ceildivui %c4096, %2 : index
      %12 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf16>
      %13 = loom.semaphore_take %12 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %14 = loom.init_tensor %13[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %15 = linalg.fill ins(%cst : f16) outs(%14 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
      %16 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %17 = loom.semaphore_take %16 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %18 = loom.init_tensor %17[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %19 = linalg.fill ins(%cst_0 : f16) outs(%18 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %20 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %21 = loom.semaphore_take %20 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %22 = loom.init_tensor %21[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %23 = linalg.fill ins(%cst_1 : f16) outs(%22 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
      %24 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf16>
      %25 = loom.semaphore_take %24 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %26 = loom.init_tensor %25[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %27 = loom.semaphore_take %24 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %28 = loom.init_tensor %27[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %29 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %30 = loom.semaphore_take %29 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %31 = loom.init_tensor %30[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %32 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %33 = loom.semaphore_take %32 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %34 = loom.init_tensor %33[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %35 = loom.alloc [%0, %1, 1] on @L1 : memref<?x?x1xf16>
      %36 = loom.semaphore_take %35 : memref<?x?x1xf16> -> memref<?x?x1xf16>
      %37 = loom.init_tensor %36[%0, %1, 1] : memref<?x?x1xf16> -> tensor<?x?x1xf16>
      %38 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf16>
      %39 = loom.semaphore_take %38 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %40 = loom.alloc [%0, %1, %2] on @L1 : memref<?x?x?xf16>
      %41 = loom.semaphore_take %40 : memref<?x?x?xf16> -> memref<?x?x?xf16>
      %42 = loom.init_tensor %41[%0, %1, %2] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
      %43 = loom.alloc [%0, %1, %2] on @L1 : memref<?x?x?xf16>
      %44 = loom.semaphore_take %43 : memref<?x?x?xf16> -> memref<?x?x?xf16>
      %45 = loom.init_tensor %44[%0, %1, %2] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
      %46 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf16>
      %47 = loom.semaphore_take %46 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %48:3 = scf.for %arg6 = %c0 to %11 step %c1 iter_args(%arg7 = %23, %arg8 = %19, %arg9 = %15) -> (tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>) {
        %56 = arith.muli %arg6, %2 : index
        %57 = loom.subview %arg0[%5, 0, %56] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x128x4096xf16> to memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>>
        loom.copy %57, %39 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[524288, 4096, 1], offset: ?>> to memref<?x128x?xf16>
        %58 = loom.bufferize_to_tensor %39[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %59 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        %60 = linalg.batch_matmul ins(%10, %58 : tensor<?x?x128xf16>, tensor<?x128x?xf16>) outs(%59 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        loom.semaphore_give %39 : memref<?x128x?xf16>
        %61 = linalg.fill ins(%cst_1 : f16) outs(%31 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%60 : tensor<?x?x?xf16>) outs(%61 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %76 = arith.maximumf %in, %out : f16
          linalg.yield %76 : f16
        } -> tensor<?x?x1xf16>
        %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %62 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%31 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %76 = arith.mulf %in_3, %cst_2 : f16
          %77 = arith.cmpf ogt, %in, %76 : f16
          %78 = arith.select %77, %in, %76 : f16
          linalg.yield %78 : f16
        } -> tensor<?x?x1xf16>
        %64 = loom.broadcast ins(%63 : tensor<?x?x1xf16>) outs(%45 : tensor<?x?x?xf16>) dim(2) -> tensor<?x?x?xf16>
        %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %64 : tensor<?x?x?xf16>, tensor<?x?x?xf16>) outs(%42 : tensor<?x?x?xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %76 = arith.mulf %in, %cst_2 : f16
          %77 = arith.subf %76, %in_3 : f16
          %78 = math.exp %77 : f16
          linalg.yield %78 : f16
        } -> tensor<?x?x?xf16>
        loom.semaphore_give %44 : memref<?x?x?xf16>
        %66 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x?x?xf16>) outs(%66 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %76 = arith.addf %in, %out : f16
          linalg.yield %76 : f16
        } -> tensor<?x?x1xf16>
        %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %63 : tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%37 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %76 = arith.subf %in, %in_3 : f16
          %77 = math.exp %76 : f16
          linalg.yield %77 : f16
        } -> tensor<?x?x1xf16>
        %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %68, %67 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x1xf16>) outs(%arg8 : tensor<?x?x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %76 = arith.mulf %in, %in_3 : f16
          %77 = arith.addf %76, %in_4 : f16
          linalg.yield %77 : f16
        } -> tensor<?x?x1xf16>
        loom.semaphore_give %33 : memref<?x?x1xf16>
        %70 = loom.subview %arg1[%5, %56, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
        loom.copy %70, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %71 = loom.bufferize_to_tensor %47[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %72 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        %73 = linalg.batch_matmul ins(%65, %71 : tensor<?x?x?xf16>, tensor<?x?x128xf16>) outs(%72 : tensor<?x?x128xf16>) -> tensor<?x?x128xf16>
        loom.semaphore_give %47 : memref<?x?x128xf16>
        loom.semaphore_give %41 : memref<?x?x?xf16>
        %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %arg9, %68 : tensor<?x?x128xf16>, tensor<?x?x128xf16>, tensor<?x?x1xf16>) outs(%arg9 : tensor<?x?x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %76 = arith.mulf %in_3, %in_4 : f16
          %77 = arith.addf %in, %76 : f16
          linalg.yield %77 : f16
        } -> tensor<?x?x128xf16>
        loom.semaphore_give %36 : memref<?x?x1xf16>
        loom.semaphore_give %27 : memref<?x?x128xf16>
        %75 = linalg.copy ins(%63 : tensor<?x?x1xf16>) outs(%arg7 : tensor<?x?x1xf16>) -> tensor<?x?x1xf16>
        loom.semaphore_give %30 : memref<?x?x1xf16>
        scf.yield %75, %69, %74 : tensor<?x?x1xf16>, tensor<?x?x1xf16>, tensor<?x?x128xf16>
      }
      loom.semaphore_give %21 : memref<?x?x1xf16>
      loom.semaphore_give %8 : memref<?x?x128xf16>
      %49 = loom.broadcast ins(%48#1 : tensor<?x?x1xf16>) outs(%26 : tensor<?x?x128xf16>) dim(2) -> tensor<?x?x128xf16>
      loom.semaphore_give %17 : memref<?x?x1xf16>
      %50 = loom.alloc [%0, %1, 128] on @L1 : memref<?x?x128xf16>
      %51 = loom.semaphore_take %50 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %52 = loom.init_tensor %51[%0, %1, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
      %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48#2, %49 : tensor<?x?x128xf16>, tensor<?x?x128xf16>) outs(%52 : tensor<?x?x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %56 = arith.divf %in, %in_3 : f16
        linalg.yield %56 : f16
      } -> tensor<?x?x128xf16>
      loom.semaphore_give %25 : memref<?x?x128xf16>
      loom.semaphore_give %13 : memref<?x?x128xf16>
      %54 = loom.subview %arg3[%5, %6, 0] [%0, %1, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<32x4096x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      %55 = loom.bufferize_to_memref %53 : tensor<?x?x128xf16> -> memref<?x?x128xf16>
      loom.copy %55, %54 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?x128xf16> to memref<?x?x128xf16, strided<[524288, 128, 1], offset: ?>>
      loom.semaphore_give %51 : memref<?x?x128xf16>
    }
    return
  }
}
