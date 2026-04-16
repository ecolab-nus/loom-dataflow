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
      %13 = arith.ceildivui %1, %2 : index
      %14 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %15 = loom.semaphore_take %14 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %16 = loom.init_tensor %15[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %17 = linalg.fill ins(%cst_0 : f16) outs(%16 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %18 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %19 = loom.semaphore_take %18 : memref<?x32xf16> -> memref<?x32xf16>
      %20 = loom.init_tensor %19[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %21 = loom.semaphore_take %18 : memref<?x32xf16> -> memref<?x32xf16>
      %22 = loom.init_tensor %21[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %23 = loom.semaphore_take %18 : memref<?x32xf16> -> memref<?x32xf16>
      %24 = loom.init_tensor %23[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %25 = loom.semaphore_take %18 : memref<?x32xf16> -> memref<?x32xf16>
      %26 = loom.init_tensor %25[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %27 = linalg.fill ins(%cst_1 : f16) outs(%26 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %28 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %29 = loom.semaphore_take %28 : memref<?x32xf16> -> memref<?x32xf16>
      %30 = loom.init_tensor %29[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %31 = linalg.fill ins(%cst_2 : f16) outs(%30 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %32 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %33 = loom.semaphore_take %32 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %34 = loom.init_tensor %33[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %35 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %36 = loom.semaphore_take %35 : memref<?x32xf16> -> memref<?x32xf16>
      %37 = loom.init_tensor %36[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %38 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %39 = loom.semaphore_take %38 : memref<?x32xf16> -> memref<?x32xf16>
      %40 = loom.init_tensor %39[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %41 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %42 = loom.semaphore_take %41 : memref<?x32xf16> -> memref<?x32xf16>
      %43 = loom.init_tensor %42[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %44 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf16>
      %45 = loom.semaphore_take %44 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %46 = loom.alloc [%0, 32, %2] on @L1 : memref<?x32x?xf16>
      %47 = loom.semaphore_take %46 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %48 = loom.init_tensor %47[%0, 32, %2] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %49 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf16>
      %50 = loom.semaphore_take %49 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %51:3 = scf.for %arg6 = %c0 to %13 step %c1 iter_args(%arg7 = %31, %arg8 = %27, %arg9 = %17) -> (tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>) {
        %60 = arith.muli %arg6, %2 : index
        %61 = arith.addi %12, %60 : index
        %62 = loom.subview %arg0[%5, 0, %61] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        loom.copy %62, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
        %63 = loom.bufferize_to_tensor %45[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %64 = linalg.fill ins(%cst_0 : f16) outs(%48 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %65 = linalg.batch_matmul ins(%11, %63 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%64 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        loom.semaphore_give %45 : memref<?x128x?xf16>
        %66 = linalg.fill ins(%cst_2 : f16) outs(%37 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%65 : tensor<?x32x?xf16>) outs(%66 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %80 = arith.maximumf %in, %out : f16
          linalg.yield %80 : f16
        } -> tensor<?x32xf16>
        %68 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %67 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%37 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %80 = arith.mulf %in_4, %cst_3 : f16
          %81 = arith.cmpf ogt, %in, %80 : f16
          %82 = arith.select %81, %in, %80 : f16
          linalg.yield %82 : f16
        } -> tensor<?x32xf16>
        %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%65, %68 : tensor<?x32x?xf16>, tensor<?x32xf16>) outs(%48 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %80 = arith.mulf %in, %cst_3 : f16
          %81 = arith.subf %80, %in_4 : f16
          %82 = math.powf %cst, %81 : f16
          linalg.yield %82 : f16
        } -> tensor<?x32x?xf16>
        %70 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %71 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%69 : tensor<?x32x?xf16>) outs(%70 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %80 = arith.addf %in, %out : f16
          linalg.yield %80 : f16
        } -> tensor<?x32xf16>
        %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %68 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%43 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %80 = arith.subf %in, %in_4 : f16
          %81 = math.powf %cst, %80 : f16
          linalg.yield %81 : f16
        } -> tensor<?x32xf16>
        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %72, %71 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32xf16>) outs(%arg8 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
          %80 = arith.mulf %in, %in_4 : f16
          %81 = arith.addf %80, %in_5 : f16
          linalg.yield %81 : f16
        } -> tensor<?x32xf16>
        loom.semaphore_give %39 : memref<?x32xf16>
        %74 = loom.subview %arg1[%5, %61, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        loom.copy %74, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %75 = loom.bufferize_to_tensor %50[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %76 = linalg.fill ins(%cst_0 : f16) outs(%34 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %77 = linalg.batch_matmul ins(%69, %75 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%76 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %50 : memref<?x?x128xf16>
        loom.semaphore_give %47 : memref<?x32x?xf16>
        %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%77, %arg9, %72 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32xf16>) outs(%arg9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
          %80 = arith.mulf %in_4, %in_5 : f16
          %81 = arith.addf %in, %80 : f16
          linalg.yield %81 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %42 : memref<?x32xf16>
        loom.semaphore_give %33 : memref<?x32x128xf16>
        %79 = linalg.copy ins(%68 : tensor<?x32xf16>) outs(%arg7 : tensor<?x32xf16>) -> tensor<?x32xf16>
        loom.semaphore_give %36 : memref<?x32xf16>
        scf.yield %79, %73, %78 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>
      }
      loom.semaphore_give %9 : memref<?x32x128xf16>
      %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%51#1, %51#0 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%24 : tensor<?x32xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %60 = math.log2 %in : f16
        %61 = arith.addf %60, %in_4 : f16
        linalg.yield %61 : f16
      } -> tensor<?x32xf16>
      loom.semaphore_give %25 : memref<?x32xf16>
      loom.semaphore_give %29 : memref<?x32xf16>
      %53 = arith.cmpi eq, %arg5, %c0 : index
      %54 = loom.alloc [%4, %0, 32] on @L1 : memref<?x?x32xf16>
      %55 = loom.semaphore_take %54 : memref<?x?x32xf16> -> memref<?x?x32xf16>
      %56 = loom.init_tensor %55[%4, %0, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
      %57 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %58 = loom.semaphore_take %57 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %59 = loom.init_tensor %58[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      scf.if %53 {
        %60 = loom.gather ins(%52 : tensor<?x32xf16>) outs(%56 : tensor<?x?x32xf16>) across(%arg5 : index) -> tensor<?x?x32xf16>
        loom.semaphore_give %23 : memref<?x32xf16>
        %61 = linalg.fill ins(%cst_2 : f16) outs(%22 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %62 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%60 : tensor<?x?x32xf16>) outs(%61 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %72 = arith.maximumf %in, %out : f16
          linalg.yield %72 : f16
        } -> tensor<?x32xf16>
        %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%60, %62 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%56 : tensor<?x?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %72 = arith.subf %in, %in_4 : f16
          %73 = math.powf %cst, %72 : f16
          linalg.yield %73 : f16
        } -> tensor<?x?x32xf16>
        loom.semaphore_give %21 : memref<?x32xf16>
        %64 = linalg.fill ins(%cst_0 : f16) outs(%20 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%63 : tensor<?x?x32xf16>) outs(%64 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %72 = arith.addf %in, %out : f16
          linalg.yield %72 : f16
        } -> tensor<?x32xf16>
        %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%63, %65 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%56 : tensor<?x?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %72 = arith.divf %in, %in_4 : f16
          linalg.yield %72 : f16
        } -> tensor<?x?x32xf16>
        loom.semaphore_give %19 : memref<?x32xf16>
        %67 = loom.gather ins(%51#2 : tensor<?x32x128xf16>) outs(%59 : tensor<?x?x32x128xf16>) across(%arg5 : index) -> tensor<?x?x32x128xf16>
        %68 = linalg.fill ins(%cst_0 : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%67, %66 : tensor<?x?x32x128xf16>, tensor<?x?x32xf16>) outs(%68 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %72 = arith.mulf %in, %in_4 : f16
          %73 = arith.addf %72, %out : f16
          linalg.yield %73 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %58 : memref<?x?x32x128xf16>
        loom.semaphore_give %55 : memref<?x?x32xf16>
        %70 = loom.subview %arg2[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %71 = loom.bufferize_to_memref %69 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
        loom.copy %71, %70 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.semaphore_give %7 : memref<?x32x128xf16>
      }
      loom.semaphore_give %15 : memref<?x32x128xf16>
    }
    return
  }
}
