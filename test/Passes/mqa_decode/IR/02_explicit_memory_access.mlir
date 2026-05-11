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
      %8 = loom.subview %arg2[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
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
      %33 = loom.semaphore_take %30 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %34 = loom.init_tensor %33[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %35 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %36 = loom.semaphore_take %35 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %37 = loom.init_tensor %36[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %38 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %39 = loom.semaphore_take %38 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %40 = loom.init_tensor %39[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %41 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %42 = loom.semaphore_take %41 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %43 = loom.init_tensor %42[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %44 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %45 = loom.semaphore_take %44 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %46 = loom.init_tensor %45[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %47 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf16>
      %48 = loom.semaphore_take %47 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %49 = loom.alloc [%0, 32, %2] on @L1 : memref<?x32x?xf16>
      %50 = loom.semaphore_take %49 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %51 = loom.init_tensor %50[%0, 32, %2] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %52 = loom.alloc [%0, 32, %2] on @L1 : memref<?x32x?xf16>
      %53 = loom.semaphore_take %52 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %54 = loom.init_tensor %53[%0, 32, %2] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %55 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf16>
      %56 = loom.semaphore_take %55 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %57:3 = scf.for %arg6 = %c0 to %11 step %c1 iter_args(%arg7 = %29, %arg8 = %25, %arg9 = %17) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %85 = arith.muli %arg6, %2 : index
        %86 = arith.addi %10, %85 : index
        %87 = loom.subview %arg0[%5, 0, %86] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        loom.copy %87, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
        %88 = loom.bufferize_to_tensor %48[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %89 = linalg.fill ins(%cst : f16) outs(%51 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %90 = linalg.batch_matmul ins(%9, %88 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%89 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        loom.semaphore_give %48 : memref<?x128x?xf16>
        %91 = linalg.fill ins(%cst_1 : f16) outs(%40 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%90 : tensor<?x32x?xf16>) outs(%91 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %107 = arith.maximumf %in, %out : f16
          linalg.yield %107 : f16
        } -> tensor<?x32x1xf16>
        %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %92 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%40 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %107 = arith.mulf %in_3, %cst_2 : f16
          %108 = arith.cmpf ogt, %in, %107 : f16
          %109 = arith.select %108, %in, %107 : f16
          linalg.yield %109 : f16
        } -> tensor<?x32x1xf16>
        %94 = loom.broadcast ins(%93 : tensor<?x32x1xf16>) outs(%54 : tensor<?x32x?xf16>) dim(2) -> tensor<?x32x?xf16>
        %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%90, %94 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%51 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %107 = arith.mulf %in, %cst_2 : f16
          %108 = arith.subf %107, %in_3 : f16
          %109 = math.exp %108 : f16
          linalg.yield %109 : f16
        } -> tensor<?x32x?xf16>
        loom.semaphore_give %53 : memref<?x32x?xf16>
        %96 = linalg.fill ins(%cst : f16) outs(%43 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%95 : tensor<?x32x?xf16>) outs(%96 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %107 = arith.addf %in, %out : f16
          linalg.yield %107 : f16
        } -> tensor<?x32x1xf16>
        %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %93 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%46 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %107 = arith.subf %in, %in_3 : f16
          %108 = math.exp %107 : f16
          linalg.yield %108 : f16
        } -> tensor<?x32x1xf16>
        %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %98, %97 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg8 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %107 = arith.mulf %in, %in_3 : f16
          %108 = arith.addf %107, %in_4 : f16
          linalg.yield %108 : f16
        } -> tensor<?x32x1xf16>
        loom.semaphore_give %42 : memref<?x32x1xf16>
        %100 = loom.broadcast ins(%98 : tensor<?x32x1xf16>) outs(%34 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
        loom.semaphore_give %45 : memref<?x32x1xf16>
        %101 = loom.subview %arg1[%5, %86, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        loom.copy %101, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %102 = loom.bufferize_to_tensor %56[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %103 = linalg.fill ins(%cst : f16) outs(%37 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %104 = linalg.batch_matmul ins(%95, %102 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%103 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %56 : memref<?x?x128xf16>
        loom.semaphore_give %50 : memref<?x32x?xf16>
        %105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%104, %arg9, %100 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %107 = arith.mulf %in_3, %in_4 : f16
          %108 = arith.addf %in, %107 : f16
          linalg.yield %108 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %36 : memref<?x32x128xf16>
        loom.semaphore_give %33 : memref<?x32x128xf16>
        %106 = linalg.copy ins(%93 : tensor<?x32x1xf16>) outs(%arg7 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        loom.semaphore_give %39 : memref<?x32x1xf16>
        scf.yield %106, %99, %105 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      loom.semaphore_give %7 : memref<?x32x128xf16>
      %58 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %59 = loom.semaphore_take %58 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %60 = loom.init_tensor %59[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %61 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57#1, %57#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%60 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %85 = math.log %in : f16
        %86 = arith.addf %85, %in_3 : f16
        linalg.yield %86 : f16
      } -> tensor<?x32x1xf16>
      loom.semaphore_give %27 : memref<?x32x1xf16>
      %62 = loom.broadcast ins(%57#1 : tensor<?x32x1xf16>) outs(%32 : tensor<?x32x128xf16>) dim(2) -> tensor<?x32x128xf16>
      loom.semaphore_give %23 : memref<?x32x1xf16>
      %63 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %64 = loom.semaphore_take %63 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %65 = loom.init_tensor %64[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57#2, %62 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%65 : tensor<?x32x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %85 = arith.divf %in, %in_3 : f16
        linalg.yield %85 : f16
      } -> tensor<?x32x128xf16>
      loom.semaphore_give %31 : memref<?x32x128xf16>
      loom.semaphore_give %15 : memref<?x32x128xf16>
      %67 = loom.bufferize_to_memref %61 : tensor<?x32x1xf16> -> memref<?x32x1xf16>
      %68 = loom.alloc [%4, %0, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %69 = loom.semaphore_take %68 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      loom.gather %67, %69 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%arg5 : index), area : [1, 1] : memref<?x32x1xf16> to memref<?x?x32x1xf16>
      loom.semaphore_give %59 : memref<?x32x1xf16>
      %70 = loom.bufferize_to_tensor %69[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %71 = loom.bufferize_to_memref %66 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
      %72 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %73 = loom.semaphore_take %72 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      loom.gather %71, %73 src_mem_space @mem_array_L1 dst_mem_space @mem_array_L1 across(%arg5 : index), area : [1, 1] : memref<?x32x128xf16> to memref<?x?x32x128xf16>
      loom.semaphore_give %64 : memref<?x32x128xf16>
      %74 = loom.bufferize_to_tensor %73[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %75 = arith.cmpi eq, %arg5, %c0 : index
      %76 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %77 = loom.semaphore_take %76 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %78 = loom.init_tensor %77[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %79 = loom.alloc [%4, %0, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %80 = loom.semaphore_take %79 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      %81 = loom.init_tensor %80[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %82 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %83 = loom.semaphore_take %82 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %84 = loom.init_tensor %83[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      scf.if %75 {
        %85 = linalg.fill ins(%cst_1 : f16) outs(%22 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%70 : tensor<?x?x32x1xf16>) outs(%85 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %98 = arith.maximumf %in, %out : f16
          linalg.yield %98 : f16
        } -> tensor<?x32x1xf16>
        %87 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%70, %86 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %98 = arith.subf %in, %in_3 : f16
          %99 = math.exp %98 : f16
          linalg.yield %99 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %69 : memref<?x?x32x1xf16>
        loom.semaphore_give %21 : memref<?x32x1xf16>
        %88 = linalg.fill ins(%cst : f16) outs(%20 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%87 : tensor<?x?x32x1xf16>) outs(%88 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %98 = arith.addf %in, %out : f16
          linalg.yield %98 : f16
        } -> tensor<?x32x1xf16>
        %90 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%87, %89 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%81 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %98 = arith.divf %in, %in_3 : f16
          linalg.yield %98 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %19 : memref<?x32x1xf16>
        %91 = loom.broadcast ins(%90 : tensor<?x?x32x1xf16>) outs(%84 : tensor<?x?x32x128xf16>) dim(3) -> tensor<?x?x32x128xf16>
        loom.semaphore_give %80 : memref<?x?x32x1xf16>
        %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%74, %91 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%84 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %98 = arith.mulf %in, %in_3 : f16
          linalg.yield %98 : f16
        } -> tensor<?x?x32x128xf16>
        loom.semaphore_give %73 : memref<?x?x32x128xf16>
        %93 = linalg.fill ins(%cst : f16) outs(%14 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %94 = linalg.copy ins(%93 : tensor<?x32x128xf16>) outs(%78 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%92 : tensor<?x?x32x128xf16>) outs(%94 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %98 = arith.addf %in, %out : f16
          linalg.yield %98 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %83 : memref<?x?x32x128xf16>
        loom.semaphore_give %13 : memref<?x32x128xf16>
        %96 = loom.subview %arg3[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %97 = loom.bufferize_to_memref %95 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
        loom.copy %97, %96 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.semaphore_give %77 : memref<?x32x128xf16>
      }
    }
    return
  }
}
