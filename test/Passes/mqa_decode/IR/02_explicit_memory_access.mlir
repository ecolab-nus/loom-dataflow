module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0xFC00 : f16
    %c1 = arith.constant 1 : index
    %cst_2 = arith.constant 8.837890e-02 : f16
    %c8192 = arith.constant 8192 : index
    %c16 = arith.constant 16 : index
    %0 = loom.sym @tile_b {upper_bound = 16 : index} : index
    %1 = loom.sym @tile_s {upper_bound = 8192 : index} : index
    %2 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %3 = arith.ceildivui %c16, %0 : index
    %4 = arith.ceildivui %c8192, %1 : index
    affine.parallel (%arg4, %arg5) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = arith.muli %arg4, %0 : index
      %6 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %7 = loom.semaphore_take %6 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %8 = loom.subview %arg3[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      loom.copy %8, %7 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
      %9 = loom.bufferize_to_tensor %7[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %10 = arith.muli %arg5, %1 : index
      %11 = arith.ceildivui %1, %2 : index
      %12 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %13 = loom.semaphore_take %12 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %14 = loom.init_tensor %13[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %15 = loom.semaphore_take %12 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %16 = loom.init_tensor %15[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %17 = linalg.fill ins(%cst : f16) outs(%16 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %18 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %19 = loom.semaphore_take %18 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %20 = loom.init_tensor %19[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %21 = loom.semaphore_take %18 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %22 = loom.init_tensor %21[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %23 = loom.semaphore_take %18 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %24 = loom.init_tensor %23[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %25 = linalg.fill ins(%cst_0 : f16) outs(%24 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %26 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %27 = loom.semaphore_take %26 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %28 = loom.init_tensor %27[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %29 = linalg.fill ins(%cst_1 : f16) outs(%28 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %30 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %31 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %32 = loom.init_tensor %31[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %33 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %34 = loom.semaphore_take %33 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %35 = loom.init_tensor %34[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %36 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %37 = loom.semaphore_take %36 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %38 = loom.init_tensor %37[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %39 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %40 = loom.semaphore_take %39 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %41 = loom.init_tensor %40[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %42 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf16>
      %43 = loom.semaphore_take %42 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %44 = loom.alloc [%0, 32, %2] on @L1 : memref<?x32x?xf16>
      %45 = loom.semaphore_take %44 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %46 = loom.init_tensor %45[%0, 32, %2] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %47 = loom.alloc [%0, 32, 32] on @L1 : memref<?x32x32xf16>
      %48 = loom.semaphore_take %47 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %49 = loom.init_tensor %48[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %50 = loom.semaphore_take %47 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %51 = loom.init_tensor %50[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %52 = loom.semaphore_take %47 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %53 = loom.init_tensor %52[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %54 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf16>
      %55 = loom.semaphore_take %54 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %56:3 = scf.for %arg6 = %c0 to %11 step %c1 iter_args(%arg7 = %29, %arg8 = %25, %arg9 = %17) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %87 = arith.muli %arg6, %2 : index
        %88 = arith.addi %10, %87 : index
        %89 = loom.subview %arg0[%5, 0, %88] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        loom.copy %89, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
        %90 = loom.bufferize_to_tensor %43[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %91 = linalg.fill ins(%cst : f16) outs(%46 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %92 = linalg.batch_matmul ins(%9, %90 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%91 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        loom.semaphore_give %43 : memref<?x128x?xf16>
        %93 = linalg.fill ins(%cst_1 : f16) outs(%35 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<?x32x?xf16>) outs(%93 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %109 = arith.maximumf %in, %out : f16
          linalg.yield %109 : f16
        } -> tensor<?x32x1xf16>
        %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %94 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%35 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %109 = arith.mulf %in_3, %cst_2 : f16
          %110 = arith.cmpf ogt, %in, %109 : f16
          %111 = arith.select %110, %in, %109 : f16
          linalg.yield %111 : f16
        } -> tensor<?x32x1xf16>
        %96 = loom.broadcast ins(%95 : tensor<?x32x1xf16>) outs(%53 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
        %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%92, %96 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%46 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %109 = arith.mulf %in, %cst_2 : f16
          %110 = arith.subf %109, %in_3 : f16
          %111 = math.exp %110 : f16
          linalg.yield %111 : f16
        } -> tensor<?x32x?xf16>
        loom.semaphore_give %52 : memref<?x32x32xf16>
        %98 = linalg.fill ins(%cst : f16) outs(%38 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%97 : tensor<?x32x?xf16>) outs(%98 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %109 = arith.addf %in, %out : f16
          linalg.yield %109 : f16
        } -> tensor<?x32x1xf16>
        %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %95 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%41 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %109 = arith.subf %in, %in_3 : f16
          %110 = math.exp %109 : f16
          linalg.yield %110 : f16
        } -> tensor<?x32x1xf16>
        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %100, %99 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg8 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %109 = arith.mulf %in, %in_3 : f16
          %110 = arith.addf %109, %in_4 : f16
          linalg.yield %110 : f16
        } -> tensor<?x32x1xf16>
        loom.semaphore_give %37 : memref<?x32x1xf16>
        %102 = loom.broadcast ins(%100 : tensor<?x32x1xf16>) outs(%51 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
        loom.semaphore_give %40 : memref<?x32x1xf16>
        %103 = loom.subview %arg1[%5, %88, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        loom.copy %103, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %104 = loom.bufferize_to_tensor %55[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %105 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %106 = linalg.batch_matmul ins(%97, %104 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%105 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %55 : memref<?x?x128xf16>
        loom.semaphore_give %45 : memref<?x32x?xf16>
        %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%106, %arg9, %102 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %109 = arith.mulf %in_3, %in_4 : f16
          %110 = arith.addf %in, %109 : f16
          linalg.yield %110 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %50 : memref<?x32x32xf16>
        loom.semaphore_give %31 : memref<?x32x128xf16>
        %108 = linalg.copy ins(%95 : tensor<?x32x1xf16>) outs(%arg7 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        loom.semaphore_give %34 : memref<?x32x1xf16>
        scf.yield %108, %101, %107 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      loom.semaphore_give %7 : memref<?x32x128xf16>
      %57 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %58 = loom.semaphore_take %57 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %59 = loom.init_tensor %58[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%56#1, %56#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%59 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %87 = math.log %in : f16
        %88 = arith.addf %87, %in_3 : f16
        linalg.yield %88 : f16
      } -> tensor<?x32x1xf16>
      loom.semaphore_give %27 : memref<?x32x1xf16>
      %61 = loom.broadcast ins(%56#1 : tensor<?x32x1xf16>) outs(%49 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
      loom.semaphore_give %23 : memref<?x32x1xf16>
      %62 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %63 = loom.semaphore_take %62 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %64 = loom.init_tensor %63[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%56#2, %61 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%64 : tensor<?x32x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %87 = arith.divf %in, %in_3 : f16
        linalg.yield %87 : f16
      } -> tensor<?x32x128xf16>
      loom.semaphore_give %48 : memref<?x32x32xf16>
      loom.semaphore_give %15 : memref<?x32x128xf16>
      %66 = loom.bufferize_to_memref %60 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
      %67 = loom.alloc [%4, %0, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %68 = loom.semaphore_take %67 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      loom.gather %66, %68 across(%arg5 : index), area : [1, 1] : memref<?x32x1xf16> to memref<?x?x32x1xf16>
      loom.semaphore_give %58 : memref<?x32x1xf16>
      %69 = loom.bufferize_to_tensor %68[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %70 = loom.bufferize_to_memref %65 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
      %71 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %72 = loom.semaphore_take %71 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      loom.gather %70, %72 across(%arg5 : index), area : [1, 1] : memref<?x32x128xf16> to memref<?x?x32x128xf16>
      loom.semaphore_give %63 : memref<?x32x128xf16>
      %73 = loom.bufferize_to_tensor %72[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %74 = arith.cmpi eq, %arg5, %c0 : index
      %75 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %76 = loom.semaphore_take %75 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %77 = loom.init_tensor %76[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %78 = loom.alloc [%4, %0, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %79 = loom.semaphore_take %78 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      %80 = loom.init_tensor %79[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %81 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %82 = loom.semaphore_take %81 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %83 = loom.init_tensor %82[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %84 = loom.alloc [%4, %0, 32, 32] on @L1 : memref<?x?x32x32xf16>
      %85 = loom.semaphore_take %84 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
      %86 = loom.init_tensor %85[%4, %0, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
      scf.if %74 {
        %87 = linalg.fill ins(%cst_1 : f16) outs(%22 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%69 : tensor<?x?x32x1xf16>) outs(%87 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %100 = arith.maximumf %in, %out : f16
          linalg.yield %100 : f16
        } -> tensor<?x32x1xf16>
        %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%69, %88 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%80 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %100 = arith.subf %in, %in_3 : f16
          %101 = math.exp %100 : f16
          linalg.yield %101 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %68 : memref<?x?x32x1xf16>
        loom.semaphore_give %21 : memref<?x32x1xf16>
        %90 = linalg.fill ins(%cst : f16) outs(%20 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%89 : tensor<?x?x32x1xf16>) outs(%90 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %100 = arith.addf %in, %out : f16
          linalg.yield %100 : f16
        } -> tensor<?x32x1xf16>
        %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%89, %91 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%80 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %100 = arith.divf %in, %in_3 : f16
          linalg.yield %100 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %19 : memref<?x32x1xf16>
        %93 = loom.broadcast ins(%92 : tensor<?x?x32x1xf16>) outs(%86 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
        loom.semaphore_give %79 : memref<?x?x32x1xf16>
        %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%73, %93 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%83 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %100 = arith.mulf %in, %in_3 : f16
          linalg.yield %100 : f16
        } -> tensor<?x?x32x128xf16>
        loom.semaphore_give %85 : memref<?x?x32x32xf16>
        loom.semaphore_give %72 : memref<?x?x32x128xf16>
        %95 = linalg.fill ins(%cst : f16) outs(%14 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %96 = linalg.copy ins(%95 : tensor<?x32x128xf16>) outs(%77 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%94 : tensor<?x?x32x128xf16>) outs(%96 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %100 = arith.addf %in, %out : f16
          linalg.yield %100 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %82 : memref<?x?x32x128xf16>
        loom.semaphore_give %13 : memref<?x32x128xf16>
        %98 = loom.subview %arg2[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %99 = loom.bufferize_to_memref %97 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
        loom.copy %99, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.semaphore_give %76 : memref<?x32x128xf16>
      }
    }
    return
  }
}
