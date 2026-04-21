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
      %8 = loom.init_tensor %7[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %9 = loom.semaphore_take %6 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %10 = loom.init_tensor %9[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %11 = loom.semaphore_take %6 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %12 = loom.subview %arg3[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      loom.copy %12, %11 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
      %13 = loom.bufferize_to_tensor %11[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %14 = arith.muli %arg5, %1 : index
      %15 = arith.ceildivui %1, %2 : index
      %16 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %17 = loom.semaphore_take %16 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %18 = loom.init_tensor %17[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %19 = linalg.fill ins(%cst : f16) outs(%18 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %20 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %21 = loom.semaphore_take %20 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %22 = loom.init_tensor %21[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %23 = loom.semaphore_take %20 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %24 = loom.init_tensor %23[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %25 = loom.semaphore_take %20 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %26 = loom.init_tensor %25[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %27 = linalg.fill ins(%cst_0 : f16) outs(%26 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %28 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %29 = loom.semaphore_take %28 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %30 = loom.init_tensor %29[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %31 = loom.semaphore_take %28 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %32 = loom.init_tensor %31[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %33 = linalg.fill ins(%cst_1 : f16) outs(%32 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %34 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %35 = loom.semaphore_take %34 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %36 = loom.init_tensor %35[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %37 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %38 = loom.semaphore_take %37 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %39 = loom.init_tensor %38[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %40 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %41 = loom.semaphore_take %40 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %42 = loom.init_tensor %41[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %43 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %45 = loom.init_tensor %44[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %46 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf16>
      %47 = loom.semaphore_take %46 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %48 = loom.alloc [%0, 32, %2] on @L1 : memref<?x32x?xf16>
      %49 = loom.semaphore_take %48 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %50 = loom.init_tensor %49[%0, 32, %2] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %51 = loom.alloc [%0, 32, 32] on @L1 : memref<?x32x32xf16>
      %52 = loom.semaphore_take %51 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %53 = loom.init_tensor %52[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %54 = loom.semaphore_take %51 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %55 = loom.init_tensor %54[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %56 = loom.semaphore_take %51 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %57 = loom.init_tensor %56[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %58 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf16>
      %59 = loom.semaphore_take %58 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %60:3 = scf.for %arg6 = %c0 to %15 step %c1 iter_args(%arg7 = %33, %arg8 = %27, %arg9 = %19) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %82 = arith.muli %arg6, %2 : index
        %83 = arith.addi %14, %82 : index
        %84 = loom.subview %arg0[%5, 0, %83] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        loom.copy %84, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
        %85 = loom.bufferize_to_tensor %47[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %86 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %87 = linalg.batch_matmul ins(%13, %85 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%86 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        loom.semaphore_give %47 : memref<?x128x?xf16>
        %88 = linalg.fill ins(%cst_1 : f16) outs(%39 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%87 : tensor<?x32x?xf16>) outs(%88 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %104 = arith.maximumf %in, %out : f16
          linalg.yield %104 : f16
        } -> tensor<?x32x1xf16>
        %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %89 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%39 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %104 = arith.mulf %in_3, %cst_2 : f16
          %105 = arith.cmpf ogt, %in, %104 : f16
          %106 = arith.select %105, %in, %104 : f16
          linalg.yield %106 : f16
        } -> tensor<?x32x1xf16>
        %91 = loom.broadcast ins(%90 : tensor<?x32x1xf16>) outs(%57 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
        %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%87, %91 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%50 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %104 = arith.mulf %in, %cst_2 : f16
          %105 = arith.subf %104, %in_3 : f16
          %106 = math.exp %105 : f16
          linalg.yield %106 : f16
        } -> tensor<?x32x?xf16>
        loom.semaphore_give %56 : memref<?x32x32xf16>
        %93 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %94 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%92 : tensor<?x32x?xf16>) outs(%93 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %104 = arith.addf %in, %out : f16
          linalg.yield %104 : f16
        } -> tensor<?x32x1xf16>
        %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %90 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%45 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %104 = arith.subf %in, %in_3 : f16
          %105 = math.exp %104 : f16
          linalg.yield %105 : f16
        } -> tensor<?x32x1xf16>
        %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %95, %94 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg8 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %104 = arith.mulf %in, %in_3 : f16
          %105 = arith.addf %104, %in_4 : f16
          linalg.yield %105 : f16
        } -> tensor<?x32x1xf16>
        loom.semaphore_give %41 : memref<?x32x1xf16>
        %97 = loom.broadcast ins(%95 : tensor<?x32x1xf16>) outs(%55 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
        loom.semaphore_give %44 : memref<?x32x1xf16>
        %98 = loom.subview %arg1[%5, %83, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        loom.copy %98, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %99 = loom.bufferize_to_tensor %59[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %100 = linalg.fill ins(%cst : f16) outs(%36 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %101 = linalg.batch_matmul ins(%92, %99 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%100 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %59 : memref<?x?x128xf16>
        loom.semaphore_give %49 : memref<?x32x?xf16>
        %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%101, %arg9, %97 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %104 = arith.mulf %in_3, %in_4 : f16
          %105 = arith.addf %in, %104 : f16
          linalg.yield %105 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %54 : memref<?x32x32xf16>
        loom.semaphore_give %35 : memref<?x32x128xf16>
        %103 = linalg.copy ins(%90 : tensor<?x32x1xf16>) outs(%arg7 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        loom.semaphore_give %38 : memref<?x32x1xf16>
        scf.yield %103, %96, %102 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      loom.semaphore_give %11 : memref<?x32x128xf16>
      %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60#1, %60#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%30 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %82 = math.log %in : f16
        %83 = arith.addf %82, %in_3 : f16
        linalg.yield %83 : f16
      } -> tensor<?x32x1xf16>
      loom.semaphore_give %31 : memref<?x32x1xf16>
      %62 = loom.broadcast ins(%60#1 : tensor<?x32x1xf16>) outs(%53 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
      loom.semaphore_give %25 : memref<?x32x1xf16>
      %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60#2, %62 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%10 : tensor<?x32x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %82 = arith.divf %in, %in_3 : f16
        linalg.yield %82 : f16
      } -> tensor<?x32x128xf16>
      loom.semaphore_give %52 : memref<?x32x32xf16>
      loom.semaphore_give %17 : memref<?x32x128xf16>
      %64 = loom.alloc [%4, %0, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %65 = loom.semaphore_take %64 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      %66 = loom.init_tensor %65[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %67 = loom.semaphore_take %28 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %68 = loom.init_tensor %67[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %69 = loom.sync ins(%61 : tensor<?x32x1xf16>) outs(%68 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %70 = loom.gather ins(%69 : tensor<?x32x1xf16>) outs(%66 : tensor<?x?x32x1xf16>) across(%arg5 : index) -> tensor<?x?x32x1xf16>
      loom.semaphore_give %67 : memref<?x32x1xf16>
      loom.semaphore_give %29 : memref<?x32x1xf16>
      %71 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %72 = loom.semaphore_take %71 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %73 = loom.init_tensor %72[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %74 = loom.semaphore_take %6 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %75 = loom.init_tensor %74[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %76 = loom.sync ins(%63 : tensor<?x32x128xf16>) outs(%75 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %77 = loom.gather ins(%76 : tensor<?x32x128xf16>) outs(%73 : tensor<?x?x32x128xf16>) across(%arg5 : index) -> tensor<?x?x32x128xf16>
      loom.semaphore_give %74 : memref<?x32x128xf16>
      loom.semaphore_give %9 : memref<?x32x128xf16>
      %78 = arith.cmpi eq, %arg5, %c0 : index
      %79 = loom.alloc [%4, %0, 32, 32] on @L1 : memref<?x?x32x32xf16>
      %80 = loom.semaphore_take %79 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
      %81 = loom.init_tensor %80[%4, %0, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
      scf.if %78 {
        %82 = linalg.fill ins(%cst_1 : f16) outs(%24 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %83 = loom.semaphore_take %64 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
        %84 = loom.init_tensor %83[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
        %85 = loom.sync ins(%70 : tensor<?x?x32x1xf16>) outs(%84 : tensor<?x?x32x1xf16>) -> tensor<?x?x32x1xf16>
        %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%85 : tensor<?x?x32x1xf16>) outs(%82 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %103 = arith.maximumf %in, %out : f16
          linalg.yield %103 : f16
        } -> tensor<?x32x1xf16>
        %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%85, %86 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%66 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %103 = arith.subf %in, %in_3 : f16
          %104 = math.exp %103 : f16
          linalg.yield %104 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %23 : memref<?x32x1xf16>
        %88 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%87 : tensor<?x?x32x1xf16>) outs(%88 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %103 = arith.addf %in, %out : f16
          linalg.yield %103 : f16
        } -> tensor<?x32x1xf16>
        %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%87, %89 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%66 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %103 = arith.divf %in, %in_3 : f16
          linalg.yield %103 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %21 : memref<?x32x1xf16>
        %91 = loom.broadcast ins(%90 : tensor<?x?x32x1xf16>) outs(%81 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
        loom.semaphore_give %83 : memref<?x?x32x1xf16>
        loom.semaphore_give %65 : memref<?x?x32x1xf16>
        %92 = loom.semaphore_take %71 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
        %93 = loom.init_tensor %92[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
        %94 = loom.sync ins(%77 : tensor<?x?x32x128xf16>) outs(%93 : tensor<?x?x32x128xf16>) -> tensor<?x?x32x128xf16>
        %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %91 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%73 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %103 = arith.mulf %in, %in_3 : f16
          linalg.yield %103 : f16
        } -> tensor<?x?x32x128xf16>
        loom.semaphore_give %80 : memref<?x?x32x32xf16>
        %96 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%95 : tensor<?x?x32x128xf16>) outs(%96 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %103 = arith.addf %in, %out : f16
          linalg.yield %103 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %92 : memref<?x?x32x128xf16>
        loom.semaphore_give %72 : memref<?x?x32x128xf16>
        %98 = loom.subview %arg2[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %99 = loom.semaphore_take %6 : memref<?x32x128xf16> -> memref<?x32x128xf16>
        %100 = loom.init_tensor %99[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
        %101 = loom.sync ins(%97 : tensor<?x32x128xf16>) outs(%100 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %102 = loom.bufferize_to_memref %101 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
        loom.copy %102, %98 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.semaphore_give %99 : memref<?x32x128xf16>
        loom.semaphore_give %7 : memref<?x32x128xf16>
      }
    }
    return
  }
}
