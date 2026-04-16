module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
  func.func @flash_decode(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
    %cst = arith.constant 2.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.000000e+00 : f16
    %cst_2 = arith.constant 0xFC00 : f16
    %cst_3 = arith.constant 1.275630e-01 : f16
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
      %13 = arith.addi %12, %1 : index
      %14 = arith.ceildivui %1, %2 : index
      %15 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %16 = loom.semaphore_take %15 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %17 = loom.init_tensor %16[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %18 = linalg.fill ins(%cst_0 : f16) outs(%17 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %19 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %20 = loom.semaphore_take %19 : memref<?x32xf16> -> memref<?x32xf16>
      %21 = loom.init_tensor %20[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %22 = loom.semaphore_take %19 : memref<?x32xf16> -> memref<?x32xf16>
      %23 = loom.init_tensor %22[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %24 = loom.semaphore_take %19 : memref<?x32xf16> -> memref<?x32xf16>
      %25 = loom.init_tensor %24[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %26 = loom.semaphore_take %19 : memref<?x32xf16> -> memref<?x32xf16>
      %27 = loom.init_tensor %26[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %28 = linalg.fill ins(%cst_1 : f16) outs(%27 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %29 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %30 = loom.semaphore_take %29 : memref<?x32xf16> -> memref<?x32xf16>
      %31 = loom.init_tensor %30[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %32 = linalg.fill ins(%cst_2 : f16) outs(%31 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %33 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %34 = loom.semaphore_take %33 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %35 = loom.init_tensor %34[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %36 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %37 = loom.semaphore_take %36 : memref<?x32xf16> -> memref<?x32xf16>
      %38 = loom.init_tensor %37[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %39 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %40 = loom.semaphore_take %39 : memref<?x32xf16> -> memref<?x32xf16>
      %41 = loom.init_tensor %40[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %42 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %43 = loom.semaphore_take %42 : memref<?x32xf16> -> memref<?x32xf16>
      %44 = loom.init_tensor %43[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %45 = loom.alloc [%0, 32, %2] on @L1 : memref<?x32x?xf16>
      %46 = loom.semaphore_take %45 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %47 = loom.init_tensor %46[%0, 32, %2] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %48:3 = scf.for %arg6 = %c0 to %14 step %c1 iter_args(%arg7 = %32, %arg8 = %28, %arg9 = %18) -> (tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>) {
        %57 = arith.muli %arg6, %2 : index
        %58 = arith.addi %12, %57 : index
        %59 = arith.addi %58, %2 : index
        %60 = arith.cmpi ult, %59, %13 : index
        %61 = arith.select %60, %59, %13 : index
        %62 = arith.subi %61, %58 : index
        %63 = loom.alloc [%0, 128, %62] on @L1 : memref<?x128x?xf16>
        %64 = loom.semaphore_take %63 : memref<?x128x?xf16> -> memref<?x128x?xf16>
        %65 = loom.subview %arg0[%5, 0, %58] [%0, 128, %62] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        loom.copy %65, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
        %66 = loom.bufferize_to_tensor %64[%0, 128, %62] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %67 = linalg.fill ins(%cst_0 : f16) outs(%47 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %68 = linalg.batch_matmul ins(%11, %66 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%67 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        loom.semaphore_give %64 : memref<?x128x?xf16>
        %69 = linalg.fill ins(%cst_2 : f16) outs(%38 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%68 : tensor<?x32x?xf16>) outs(%69 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %85 = arith.maximumf %in, %out : f16
          linalg.yield %85 : f16
        } -> tensor<?x32xf16>
        %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %70 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%38 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %85 = arith.mulf %in_4, %cst_3 : f16
          %86 = arith.cmpf ogt, %in, %85 : f16
          %87 = arith.select %86, %in, %85 : f16
          linalg.yield %87 : f16
        } -> tensor<?x32xf16>
        %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%68, %71 : tensor<?x32x?xf16>, tensor<?x32xf16>) outs(%47 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %85 = arith.mulf %in, %cst_3 : f16
          %86 = arith.subf %85, %in_4 : f16
          %87 = math.powf %cst, %86 : f16
          linalg.yield %87 : f16
        } -> tensor<?x32x?xf16>
        %73 = linalg.fill ins(%cst_0 : f16) outs(%41 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%72 : tensor<?x32x?xf16>) outs(%73 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %85 = arith.addf %in, %out : f16
          linalg.yield %85 : f16
        } -> tensor<?x32xf16>
        %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %71 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%44 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %85 = arith.subf %in, %in_4 : f16
          %86 = math.powf %cst, %85 : f16
          linalg.yield %86 : f16
        } -> tensor<?x32xf16>
        %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %75, %74 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32xf16>) outs(%arg8 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
          %85 = arith.mulf %in, %in_4 : f16
          %86 = arith.addf %85, %in_5 : f16
          linalg.yield %86 : f16
        } -> tensor<?x32xf16>
        loom.semaphore_give %40 : memref<?x32xf16>
        %77 = loom.alloc [%0, %62, 128] on @L1 : memref<?x?x128xf16>
        %78 = loom.semaphore_take %77 : memref<?x?x128xf16> -> memref<?x?x128xf16>
        %79 = loom.subview %arg1[%5, %58, 0] [%0, %62, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        loom.copy %79, %78 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %80 = loom.bufferize_to_tensor %78[%0, %62, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %81 = linalg.fill ins(%cst_0 : f16) outs(%35 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %82 = linalg.batch_matmul ins(%72, %80 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%81 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %78 : memref<?x?x128xf16>
        loom.semaphore_give %46 : memref<?x32x?xf16>
        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%82, %arg9, %75 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32xf16>) outs(%arg9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
          %85 = arith.mulf %in_4, %in_5 : f16
          %86 = arith.addf %in, %85 : f16
          linalg.yield %86 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %43 : memref<?x32xf16>
        loom.semaphore_give %34 : memref<?x32x128xf16>
        %84 = linalg.copy ins(%71 : tensor<?x32xf16>) outs(%arg7 : tensor<?x32xf16>) -> tensor<?x32xf16>
        loom.semaphore_give %37 : memref<?x32xf16>
        scf.yield %84, %76, %83 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>
      }
      loom.semaphore_give %9 : memref<?x32x128xf16>
      %49 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%48#1, %48#0 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%25 : tensor<?x32xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %57 = math.log2 %in : f16
        %58 = arith.addf %57, %in_4 : f16
        linalg.yield %58 : f16
      } -> tensor<?x32xf16>
      loom.semaphore_give %26 : memref<?x32xf16>
      loom.semaphore_give %30 : memref<?x32xf16>
      %50 = arith.cmpi eq, %arg5, %c0 : index
      %51 = loom.alloc [%4, %0, 32] on @L1 : memref<?x?x32xf16>
      %52 = loom.semaphore_take %51 : memref<?x?x32xf16> -> memref<?x?x32xf16>
      %53 = loom.init_tensor %52[%4, %0, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
      %54 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %55 = loom.semaphore_take %54 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %56 = loom.init_tensor %55[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      scf.if %50 {
        %57 = loom.gather ins(%49 : tensor<?x32xf16>) outs(%53 : tensor<?x?x32xf16>) across(%arg5 : index) -> tensor<?x?x32xf16>
        loom.semaphore_give %24 : memref<?x32xf16>
        %58 = linalg.fill ins(%cst_2 : f16) outs(%23 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %59 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%57 : tensor<?x?x32xf16>) outs(%58 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %69 = arith.maximumf %in, %out : f16
          linalg.yield %69 : f16
        } -> tensor<?x32xf16>
        %60 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%57, %59 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%53 : tensor<?x?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %69 = arith.subf %in, %in_4 : f16
          %70 = math.powf %cst, %69 : f16
          linalg.yield %70 : f16
        } -> tensor<?x?x32xf16>
        loom.semaphore_give %22 : memref<?x32xf16>
        %61 = linalg.fill ins(%cst_0 : f16) outs(%21 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%60 : tensor<?x?x32xf16>) outs(%61 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %69 = arith.addf %in, %out : f16
          linalg.yield %69 : f16
        } -> tensor<?x32xf16>
        %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %62 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%53 : tensor<?x?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %69 = arith.divf %in, %in_4 : f16
          linalg.yield %69 : f16
        } -> tensor<?x?x32xf16>
        loom.semaphore_give %20 : memref<?x32xf16>
        %64 = loom.gather ins(%48#2 : tensor<?x32x128xf16>) outs(%56 : tensor<?x?x32x128xf16>) across(%arg5 : index) -> tensor<?x?x32x128xf16>
        %65 = linalg.fill ins(%cst_0 : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%64, %63 : tensor<?x?x32x128xf16>, tensor<?x?x32xf16>) outs(%65 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %69 = arith.mulf %in, %in_4 : f16
          %70 = arith.addf %69, %out : f16
          linalg.yield %70 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %55 : memref<?x?x32x128xf16>
        loom.semaphore_give %52 : memref<?x?x32xf16>
        %67 = loom.subview %arg2[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %68 = loom.bufferize_to_memref %66 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
        loom.copy %68, %67 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.semaphore_give %7 : memref<?x32x128xf16>
      }
      loom.semaphore_give %16 : memref<?x32x128xf16>
    }
    return
  }
}
