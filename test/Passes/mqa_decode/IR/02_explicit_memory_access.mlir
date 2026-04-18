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
        %68 = arith.muli %arg6, %2 : index
        %69 = arith.addi %12, %68 : index
        %70 = loom.subview %arg0[%5, 0, %69] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        loom.copy %70, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
        %71 = loom.bufferize_to_tensor %45[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %72 = linalg.fill ins(%cst_0 : f16) outs(%48 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %73 = linalg.batch_matmul ins(%11, %71 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%72 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        loom.semaphore_give %45 : memref<?x128x?xf16>
        %74 = linalg.fill ins(%cst_2 : f16) outs(%37 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %75 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%73 : tensor<?x32x?xf16>) outs(%74 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %88 = arith.maximumf %in, %out : f16
          linalg.yield %88 : f16
        } -> tensor<?x32xf16>
        %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %75 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%37 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %88 = arith.mulf %in_4, %cst_3 : f16
          %89 = arith.cmpf ogt, %in, %88 : f16
          %90 = arith.select %89, %in, %88 : f16
          linalg.yield %90 : f16
        } -> tensor<?x32xf16>
        %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%73, %76 : tensor<?x32x?xf16>, tensor<?x32xf16>) outs(%48 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %88 = arith.mulf %in, %cst_3 : f16
          %89 = arith.subf %88, %in_4 : f16
          %90 = math.powf %cst, %89 : f16
          linalg.yield %90 : f16
        } -> tensor<?x32x?xf16>
        %78 = linalg.fill ins(%cst_0 : f16) outs(%40 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %79 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%77 : tensor<?x32x?xf16>) outs(%78 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %88 = arith.addf %in, %out : f16
          linalg.yield %88 : f16
        } -> tensor<?x32xf16>
        %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg7, %76 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%43 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %88 = arith.subf %in, %in_4 : f16
          %89 = math.powf %cst, %88 : f16
          linalg.yield %89 : f16
        } -> tensor<?x32xf16>
        %81 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg8, %80, %79 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32xf16>) outs(%arg8 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
          %88 = arith.mulf %in, %in_4 : f16
          %89 = arith.addf %88, %in_5 : f16
          linalg.yield %89 : f16
        } -> tensor<?x32xf16>
        loom.semaphore_give %39 : memref<?x32xf16>
        %82 = loom.subview %arg1[%5, %69, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        loom.copy %82, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %83 = loom.bufferize_to_tensor %50[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %84 = linalg.fill ins(%cst_0 : f16) outs(%34 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %85 = linalg.batch_matmul ins(%77, %83 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%84 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %50 : memref<?x?x128xf16>
        loom.semaphore_give %47 : memref<?x32x?xf16>
        %86 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%85, %arg9, %80 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32xf16>) outs(%arg9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %in_5: f16, %out: f16):
          %88 = arith.mulf %in_4, %in_5 : f16
          %89 = arith.addf %in, %88 : f16
          linalg.yield %89 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %42 : memref<?x32xf16>
        loom.semaphore_give %33 : memref<?x32x128xf16>
        %87 = linalg.copy ins(%76 : tensor<?x32xf16>) outs(%arg7 : tensor<?x32xf16>) -> tensor<?x32xf16>
        loom.semaphore_give %36 : memref<?x32xf16>
        scf.yield %87, %81, %86 : tensor<?x32xf16>, tensor<?x32xf16>, tensor<?x32x128xf16>
      }
      loom.semaphore_give %9 : memref<?x32x128xf16>
      %52 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%51#1, %51#0 : tensor<?x32xf16>, tensor<?x32xf16>) outs(%24 : tensor<?x32xf16>) {
      ^bb0(%in: f16, %in_4: f16, %out: f16):
        %68 = math.log2 %in : f16
        %69 = arith.addf %68, %in_4 : f16
        linalg.yield %69 : f16
      } -> tensor<?x32xf16>
      loom.semaphore_give %25 : memref<?x32xf16>
      loom.semaphore_give %29 : memref<?x32xf16>
      %53 = loom.alloc [%4, %0, 32] on @L1 : memref<?x?x32xf16>
      %54 = loom.semaphore_take %53 : memref<?x?x32xf16> -> memref<?x?x32xf16>
      %55 = loom.init_tensor %54[%4, %0, 32] : memref<?x?x32xf16> -> tensor<?x?x32xf16>
      %56 = loom.semaphore_take %18 : memref<?x32xf16> -> memref<?x32xf16>
      %57 = loom.init_tensor %56[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %58 = loom.sync ins(%52 : tensor<?x32xf16>) outs(%57 : tensor<?x32xf16>) -> tensor<?x32xf16>
      %59 = loom.gather ins(%58 : tensor<?x32xf16>) outs(%55 : tensor<?x?x32xf16>) across(%arg5 : index) -> tensor<?x?x32xf16>
      loom.semaphore_give %56 : memref<?x32xf16>
      loom.semaphore_give %23 : memref<?x32xf16>
      %60 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %61 = loom.semaphore_take %60 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %62 = loom.init_tensor %61[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %63 = loom.semaphore_take %14 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %64 = loom.init_tensor %63[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %65 = loom.sync ins(%51#2 : tensor<?x32x128xf16>) outs(%64 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %66 = loom.gather ins(%65 : tensor<?x32x128xf16>) outs(%62 : tensor<?x?x32x128xf16>) across(%arg5 : index) -> tensor<?x?x32x128xf16>
      loom.semaphore_give %63 : memref<?x32x128xf16>
      loom.semaphore_give %15 : memref<?x32x128xf16>
      %67 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %67 {
        %68 = linalg.fill ins(%cst_2 : f16) outs(%22 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%59 : tensor<?x?x32xf16>) outs(%68 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %79 = arith.maximumf %in, %out : f16
          linalg.yield %79 : f16
        } -> tensor<?x32xf16>
        %70 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%59, %69 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%55 : tensor<?x?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %79 = arith.subf %in, %in_4 : f16
          %80 = math.powf %cst, %79 : f16
          linalg.yield %80 : f16
        } -> tensor<?x?x32xf16>
        loom.semaphore_give %21 : memref<?x32xf16>
        %71 = linalg.fill ins(%cst_0 : f16) outs(%20 : tensor<?x32xf16>) -> tensor<?x32xf16>
        %72 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%70 : tensor<?x?x32xf16>) outs(%71 : tensor<?x32xf16>) {
        ^bb0(%in: f16, %out: f16):
          %79 = arith.addf %in, %out : f16
          linalg.yield %79 : f16
        } -> tensor<?x32xf16>
        %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%70, %72 : tensor<?x?x32xf16>, tensor<?x32xf16>) outs(%55 : tensor<?x?x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %79 = arith.divf %in, %in_4 : f16
          linalg.yield %79 : f16
        } -> tensor<?x?x32xf16>
        loom.semaphore_give %19 : memref<?x32xf16>
        %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%66, %73 : tensor<?x?x32x128xf16>, tensor<?x?x32xf16>) outs(%62 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %79 = arith.mulf %in, %in_4 : f16
          linalg.yield %79 : f16
        } -> tensor<?x?x32x128xf16>
        loom.semaphore_give %54 : memref<?x?x32xf16>
        %75 = linalg.fill ins(%cst_0 : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%74 : tensor<?x?x32x128xf16>) outs(%75 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %79 = arith.addf %in, %out : f16
          linalg.yield %79 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %61 : memref<?x?x32x128xf16>
        %77 = loom.subview %arg2[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %78 = loom.bufferize_to_memref %76 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
        loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.semaphore_give %7 : memref<?x32x128xf16>
      }
    }
    return
  }
}
