module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
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
      %10 = loom.subview %arg3[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
      loom.copy %10, %9 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<?x32x128xf16>
      %11 = loom.bufferize_to_tensor %9[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %12 = arith.muli %arg5, %1 : index
      %13 = arith.ceildivui %1, %2 : index
      %14 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %15 = loom.semaphore_take %14 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %16 = loom.init_tensor %15[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %17 = linalg.fill ins(%cst : f16) outs(%16 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %18 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %19 = loom.semaphore_take %18 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %20 = loom.init_tensor %19[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %21 = loom.semaphore_take %18 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %22 = loom.init_tensor %21[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %23 = loom.semaphore_take %18 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %24 = loom.init_tensor %23[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %25 = loom.semaphore_take %18 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %26 = loom.init_tensor %25[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %27 = linalg.fill ins(%cst_0 : f16) outs(%26 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %28 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %29 = loom.semaphore_take %28 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %30 = loom.init_tensor %29[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %31 = linalg.fill ins(%cst_1 : f16) outs(%30 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %32 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %33 = loom.semaphore_take %32 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %34 = loom.init_tensor %33[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %35 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %36 = loom.semaphore_take %35 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %37 = loom.init_tensor %36[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %38 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %39 = loom.semaphore_take %38 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %40 = loom.init_tensor %39[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %41 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %42 = loom.semaphore_take %41 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %43 = loom.init_tensor %42[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %44 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf16>
      %45 = loom.semaphore_take %44 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %46 = loom.alloc [%0, 32, %2] on @L1 : memref<?x32x?xf16>
      %47 = loom.semaphore_take %46 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %48 = loom.init_tensor %47[%0, 32, %2] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %49 = loom.alloc [%0, 32, 32] on @L1 : memref<?x32x32xf16>
      %50 = loom.semaphore_take %49 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %51 = loom.init_tensor %50[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %52 = loom.semaphore_take %49 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %53 = loom.init_tensor %52[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %54 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf16>
      %55 = loom.semaphore_take %54 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %56:3 = scf.for %arg6 = %c0 to %13 step %c1 iter_args(%arg7 = %31, %arg8 = %27, %arg9 = %17) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %76 = arith.muli %arg6, %2 : index
        %77 = arith.addi %12, %76 : index
        %78 = loom.subview %arg0[%5, 0, %77] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        loom.copy %78, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
        %79 = loom.bufferize_to_tensor %45[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %80 = linalg.fill ins(%cst : f16) outs(%48 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %81 = linalg.batch_matmul ins(%11, %79 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%80 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        loom.semaphore_give %45 : memref<?x128x?xf16>
        %82 = linalg.fill ins(%cst_1 : f16) outs(%37 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%81 : tensor<?x32x?xf16>) outs(%82 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %98 = arith.maximumf %in, %out : f16
          linalg.yield %98 : f16
        } -> tensor<?x32x1xf16>
        %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %83 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%37 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %98 = arith.mulf %in_3, %cst_2 : f16
          %99 = arith.cmpf ogt, %in, %98 : f16
          %100 = arith.select %99, %in, %98 : f16
          linalg.yield %100 : f16
        } -> tensor<?x32x1xf16>
        %85 = loom.broadcast ins(%84 : tensor<?x32x1xf16>) outs(%53 : tensor<?x32x32xf16>) dim(%c2 : index) -> tensor<?x32x?xf16>
        %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%81, %85 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%48 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %98 = arith.mulf %in, %cst_2 : f16
          %99 = arith.subf %98, %in_3 : f16
          %100 = math.exp %99 : f16
          linalg.yield %100 : f16
        } -> tensor<?x32x?xf16>
        loom.semaphore_give %52 : memref<?x32x32xf16>
        %87 = linalg.fill ins(%cst : f16) outs(%40 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%86 : tensor<?x32x?xf16>) outs(%87 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %98 = arith.addf %in, %out : f16
          linalg.yield %98 : f16
        } -> tensor<?x32x1xf16>
        %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %84 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%43 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %98 = arith.subf %in, %in_3 : f16
          %99 = math.exp %98 : f16
          linalg.yield %99 : f16
        } -> tensor<?x32x1xf16>
        %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %89, %88 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg8 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %98 = arith.mulf %in, %in_3 : f16
          %99 = arith.addf %98, %in_4 : f16
          linalg.yield %99 : f16
        } -> tensor<?x32x1xf16>
        loom.semaphore_give %39 : memref<?x32x1xf16>
        %91 = loom.broadcast ins(%89 : tensor<?x32x1xf16>) outs(%51 : tensor<?x32x32xf16>) dim(%c2 : index) -> tensor<?x32x128xf16>
        loom.semaphore_give %42 : memref<?x32x1xf16>
        %92 = loom.subview %arg1[%5, %77, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        loom.copy %92, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %93 = loom.bufferize_to_tensor %55[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %94 = linalg.fill ins(%cst : f16) outs(%34 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %95 = linalg.batch_matmul ins(%86, %93 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%94 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %55 : memref<?x?x128xf16>
        loom.semaphore_give %47 : memref<?x32x?xf16>
        %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %arg9, %91 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %98 = arith.mulf %in_3, %in_4 : f16
          %99 = arith.addf %in, %98 : f16
          linalg.yield %99 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %50 : memref<?x32x32xf16>
        loom.semaphore_give %33 : memref<?x32x128xf16>
        %97 = linalg.copy ins(%84 : tensor<?x32x1xf16>) outs(%arg7 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        loom.semaphore_give %36 : memref<?x32x1xf16>
        scf.yield %97, %90, %96 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      loom.semaphore_give %9 : memref<?x32x128xf16>
      %57 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%56#1, %56#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%24 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %76 = math.log %in : f16
        %77 = arith.addf %76, %in_3 : f16
        linalg.yield %77 : f16
      } -> tensor<?x32x1xf16>
      loom.semaphore_give %25 : memref<?x32x1xf16>
      loom.semaphore_give %29 : memref<?x32x1xf16>
      %58 = loom.alloc [%4, %0, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %59 = loom.semaphore_take %58 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      %60 = loom.init_tensor %59[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %61 = loom.semaphore_take %18 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %62 = loom.init_tensor %61[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %63 = loom.sync ins(%57 : tensor<?x32x1xf16>) outs(%62 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %64 = loom.gather ins(%63 : tensor<?x32x1xf16>) outs(%60 : tensor<?x?x32x1xf16>) across(%arg5 : index) -> tensor<?x?x32x1xf16>
      loom.semaphore_give %61 : memref<?x32x1xf16>
      loom.semaphore_give %23 : memref<?x32x1xf16>
      %65 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %66 = loom.semaphore_take %65 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %67 = loom.init_tensor %66[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %68 = loom.semaphore_take %14 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %69 = loom.init_tensor %68[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %70 = loom.sync ins(%56#2 : tensor<?x32x128xf16>) outs(%69 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %71 = loom.gather ins(%70 : tensor<?x32x128xf16>) outs(%67 : tensor<?x?x32x128xf16>) across(%arg5 : index) -> tensor<?x?x32x128xf16>
      loom.semaphore_give %68 : memref<?x32x128xf16>
      loom.semaphore_give %15 : memref<?x32x128xf16>
      %72 = arith.cmpi eq, %arg5, %c0 : index
      %73 = loom.alloc [%1, %0, 32, 32] on @L1 : memref<?x?x32x32xf16>
      %74 = loom.semaphore_take %73 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
      %75 = loom.init_tensor %74[%1, %0, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
      scf.if %72 {
        %76 = linalg.fill ins(%cst_1 : f16) outs(%22 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %77 = loom.semaphore_take %58 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
        %78 = loom.init_tensor %77[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
        %79 = loom.sync ins(%64 : tensor<?x?x32x1xf16>) outs(%78 : tensor<?x?x32x1xf16>) -> tensor<?x?x32x1xf16>
        %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%79 : tensor<?x?x32x1xf16>) outs(%76 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %97 = arith.maximumf %in, %out : f16
          linalg.yield %97 : f16
        } -> tensor<?x32x1xf16>
        %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%79, %80 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%60 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %97 = arith.subf %in, %in_3 : f16
          %98 = math.exp %97 : f16
          linalg.yield %98 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %21 : memref<?x32x1xf16>
        %82 = linalg.fill ins(%cst : f16) outs(%20 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%81 : tensor<?x?x32x1xf16>) outs(%82 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %97 = arith.addf %in, %out : f16
          linalg.yield %97 : f16
        } -> tensor<?x32x1xf16>
        %84 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%81, %83 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%60 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %97 = arith.divf %in, %in_3 : f16
          linalg.yield %97 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %19 : memref<?x32x1xf16>
        %85 = loom.broadcast ins(%84 : tensor<?x?x32x1xf16>) outs(%75 : tensor<?x?x32x32xf16>) dim(%c3 : index) -> tensor<?x?x32x128xf16>
        loom.semaphore_give %77 : memref<?x?x32x1xf16>
        loom.semaphore_give %59 : memref<?x?x32x1xf16>
        %86 = loom.semaphore_take %65 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
        %87 = loom.init_tensor %86[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
        %88 = loom.sync ins(%71 : tensor<?x?x32x128xf16>) outs(%87 : tensor<?x?x32x128xf16>) -> tensor<?x?x32x128xf16>
        %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%88, %85 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%67 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %97 = arith.mulf %in, %in_3 : f16
          linalg.yield %97 : f16
        } -> tensor<?x?x32x128xf16>
        loom.semaphore_give %74 : memref<?x?x32x32xf16>
        %90 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %91 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%89 : tensor<?x?x32x128xf16>) outs(%90 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %97 = arith.addf %in, %out : f16
          linalg.yield %97 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %86 : memref<?x?x32x128xf16>
        loom.semaphore_give %66 : memref<?x?x32x128xf16>
        %92 = loom.subview %arg2[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %93 = loom.semaphore_take %6 : memref<?x32x128xf16> -> memref<?x32x128xf16>
        %94 = loom.init_tensor %93[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
        %95 = loom.sync ins(%91 : tensor<?x32x128xf16>) outs(%94 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %96 = loom.bufferize_to_memref %95 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
        loom.copy %96, %92 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.semaphore_give %93 : memref<?x32x128xf16>
        loom.semaphore_give %7 : memref<?x32x128xf16>
      }
    }
    return
  }
}
