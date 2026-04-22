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
      %19 = loom.semaphore_take %16 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %20 = loom.init_tensor %19[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %21 = loom.semaphore_take %16 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %22 = loom.init_tensor %21[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      %24 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %25 = loom.semaphore_take %24 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %26 = loom.init_tensor %25[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %27 = loom.semaphore_take %24 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %28 = loom.init_tensor %27[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %29 = loom.semaphore_take %24 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %30 = loom.init_tensor %29[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %31 = loom.semaphore_take %24 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %32 = loom.init_tensor %31[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %33 = linalg.fill ins(%cst_0 : f16) outs(%32 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %34 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %35 = loom.semaphore_take %34 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %36 = loom.init_tensor %35[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %37 = loom.semaphore_take %34 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %38 = loom.init_tensor %37[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %39 = linalg.fill ins(%cst_1 : f16) outs(%38 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      %40 = loom.alloc [%0, 32, 128] on @L1 : memref<?x32x128xf16>
      %41 = loom.semaphore_take %40 : memref<?x32x128xf16> -> memref<?x32x128xf16>
      %42 = loom.init_tensor %41[%0, 32, 128] : memref<?x32x128xf16> -> tensor<?x32x128xf16>
      %43 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %44 = loom.semaphore_take %43 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %45 = loom.init_tensor %44[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %46 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %47 = loom.semaphore_take %46 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %48 = loom.init_tensor %47[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %49 = loom.alloc [%0, 32, 1] on @L1 : memref<?x32x1xf16>
      %50 = loom.semaphore_take %49 : memref<?x32x1xf16> -> memref<?x32x1xf16>
      %51 = loom.init_tensor %50[%0, 32, 1] : memref<?x32x1xf16> -> tensor<?x32x1xf16>
      %52 = loom.alloc [%0, 128, %2] on @L1 : memref<?x128x?xf16>
      %53 = loom.semaphore_take %52 : memref<?x128x?xf16> -> memref<?x128x?xf16>
      %54 = loom.alloc [%0, 32, %2] on @L1 : memref<?x32x?xf16>
      %55 = loom.semaphore_take %54 : memref<?x32x?xf16> -> memref<?x32x?xf16>
      %56 = loom.init_tensor %55[%0, 32, %2] : memref<?x32x?xf16> -> tensor<?x32x?xf16>
      %57 = loom.alloc [%0, 32, 32] on @L1 : memref<?x32x32xf16>
      %58 = loom.semaphore_take %57 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %59 = loom.init_tensor %58[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %60 = loom.semaphore_take %57 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %61 = loom.init_tensor %60[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %62 = loom.semaphore_take %57 : memref<?x32x32xf16> -> memref<?x32x32xf16>
      %63 = loom.init_tensor %62[%0, 32, 32] : memref<?x32x32xf16> -> tensor<?x32x32xf16>
      %64 = loom.alloc [%0, %2, 128] on @L1 : memref<?x?x128xf16>
      %65 = loom.semaphore_take %64 : memref<?x?x128xf16> -> memref<?x?x128xf16>
      %66:3 = scf.for %arg6 = %c0 to %15 step %c1 iter_args(%arg7 = %39, %arg8 = %33, %arg9 = %23) -> (tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>) {
        %90 = arith.muli %arg6, %2 : index
        %91 = arith.addi %14, %90 : index
        %92 = loom.subview %arg0[%5, 0, %91] [%0, 128, %2] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x128x8192xf16> to memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>>
        loom.copy %92, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x128x?xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<?x128x?xf16>
        %93 = loom.bufferize_to_tensor %53[%0, 128, %2] : memref<?x128x?xf16> -> tensor<?x128x?xf16>
        %94 = linalg.fill ins(%cst : f16) outs(%56 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        %95 = linalg.batch_matmul ins(%13, %93 : tensor<?x32x128xf16>, tensor<?x128x?xf16>) outs(%94 : tensor<?x32x?xf16>) -> tensor<?x32x?xf16>
        loom.semaphore_give %53 : memref<?x128x?xf16>
        %96 = linalg.fill ins(%cst_1 : f16) outs(%45 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%95 : tensor<?x32x?xf16>) outs(%96 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %112 = arith.maximumf %in, %out : f16
          linalg.yield %112 : f16
        } -> tensor<?x32x1xf16>
        %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %97 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%45 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %112 = arith.mulf %in_3, %cst_2 : f16
          %113 = arith.cmpf ogt, %in, %112 : f16
          %114 = arith.select %113, %in, %112 : f16
          linalg.yield %114 : f16
        } -> tensor<?x32x1xf16>
        %99 = loom.broadcast ins(%98 : tensor<?x32x1xf16>) outs(%63 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x?xf16>
        %100 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%95, %99 : tensor<?x32x?xf16>, tensor<?x32x?xf16>) outs(%56 : tensor<?x32x?xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %112 = arith.mulf %in, %cst_2 : f16
          %113 = arith.subf %112, %in_3 : f16
          %114 = math.exp %113 : f16
          linalg.yield %114 : f16
        } -> tensor<?x32x?xf16>
        loom.semaphore_give %62 : memref<?x32x32xf16>
        %101 = linalg.fill ins(%cst : f16) outs(%48 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%100 : tensor<?x32x?xf16>) outs(%101 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %112 = arith.addf %in, %out : f16
          linalg.yield %112 : f16
        } -> tensor<?x32x1xf16>
        %103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg7, %98 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%51 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %112 = arith.subf %in, %in_3 : f16
          %113 = math.exp %112 : f16
          linalg.yield %113 : f16
        } -> tensor<?x32x1xf16>
        %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg8, %103, %102 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%arg8 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %112 = arith.mulf %in, %in_3 : f16
          %113 = arith.addf %112, %in_4 : f16
          linalg.yield %113 : f16
        } -> tensor<?x32x1xf16>
        loom.semaphore_give %47 : memref<?x32x1xf16>
        %105 = loom.broadcast ins(%103 : tensor<?x32x1xf16>) outs(%61 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
        loom.semaphore_give %50 : memref<?x32x1xf16>
        %106 = loom.subview %arg1[%5, %91, 0] [%0, %2, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x8192x128xf16> to memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>>
        loom.copy %106, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<?x?x128xf16>
        %107 = loom.bufferize_to_tensor %65[%0, %2, 128] : memref<?x?x128xf16> -> tensor<?x?x128xf16>
        %108 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %109 = linalg.batch_matmul ins(%100, %107 : tensor<?x32x?xf16>, tensor<?x?x128xf16>) outs(%108 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %65 : memref<?x?x128xf16>
        loom.semaphore_give %55 : memref<?x32x?xf16>
        %110 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%109, %arg9, %105 : tensor<?x32x128xf16>, tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%arg9 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %in_4: f16, %out: f16):
          %112 = arith.mulf %in_3, %in_4 : f16
          %113 = arith.addf %in, %112 : f16
          linalg.yield %113 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %60 : memref<?x32x32xf16>
        loom.semaphore_give %41 : memref<?x32x128xf16>
        %111 = linalg.copy ins(%98 : tensor<?x32x1xf16>) outs(%arg7 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        loom.semaphore_give %44 : memref<?x32x1xf16>
        scf.yield %111, %104, %110 : tensor<?x32x1xf16>, tensor<?x32x1xf16>, tensor<?x32x128xf16>
      }
      loom.semaphore_give %11 : memref<?x32x128xf16>
      %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%66#1, %66#0 : tensor<?x32x1xf16>, tensor<?x32x1xf16>) outs(%36 : tensor<?x32x1xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %90 = math.log %in : f16
        %91 = arith.addf %90, %in_3 : f16
        linalg.yield %91 : f16
      } -> tensor<?x32x1xf16>
      loom.semaphore_give %37 : memref<?x32x1xf16>
      %68 = loom.broadcast ins(%66#1 : tensor<?x32x1xf16>) outs(%59 : tensor<?x32x32xf16>) dim(2) -> tensor<?x32x128xf16>
      loom.semaphore_give %31 : memref<?x32x1xf16>
      %69 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%66#2, %68 : tensor<?x32x128xf16>, tensor<?x32x128xf16>) outs(%10 : tensor<?x32x128xf16>) {
      ^bb0(%in: f16, %in_3: f16, %out: f16):
        %90 = arith.divf %in, %in_3 : f16
        linalg.yield %90 : f16
      } -> tensor<?x32x128xf16>
      loom.semaphore_give %58 : memref<?x32x32xf16>
      loom.semaphore_give %21 : memref<?x32x128xf16>
      %70 = loom.sync ins(%67 : tensor<?x32x1xf16>) outs(%30 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
      loom.semaphore_give %35 : memref<?x32x1xf16>
      %71 = loom.alloc [%4, %0, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %72 = loom.semaphore_take %71 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      %73 = loom.init_tensor %72[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %74 = loom.gather ins(%70 : tensor<?x32x1xf16>) outs(%73 : tensor<?x?x32x1xf16>) across(%arg5 : index) -> tensor<?x?x32x1xf16>
      loom.semaphore_give %29 : memref<?x32x1xf16>
      %75 = loom.sync ins(%69 : tensor<?x32x128xf16>) outs(%20 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
      loom.semaphore_give %9 : memref<?x32x128xf16>
      %76 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %77 = loom.semaphore_take %76 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %78 = loom.init_tensor %77[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %79 = loom.gather ins(%75 : tensor<?x32x128xf16>) outs(%78 : tensor<?x?x32x128xf16>) across(%arg5 : index) -> tensor<?x?x32x128xf16>
      loom.semaphore_give %19 : memref<?x32x128xf16>
      %80 = arith.cmpi eq, %arg5, %c0 : index
      %81 = loom.alloc [%4, %0, 32, 1] on @L1 : memref<?x?x32x1xf16>
      %82 = loom.semaphore_take %81 : memref<?x?x32x1xf16> -> memref<?x?x32x1xf16>
      %83 = loom.init_tensor %82[%4, %0, 32, 1] : memref<?x?x32x1xf16> -> tensor<?x?x32x1xf16>
      %84 = loom.alloc [%4, %0, 32, 128] on @L1 : memref<?x?x32x128xf16>
      %85 = loom.semaphore_take %84 : memref<?x?x32x128xf16> -> memref<?x?x32x128xf16>
      %86 = loom.init_tensor %85[%4, %0, 32, 128] : memref<?x?x32x128xf16> -> tensor<?x?x32x128xf16>
      %87 = loom.alloc [%4, %0, 32, 32] on @L1 : memref<?x?x32x32xf16>
      %88 = loom.semaphore_take %87 : memref<?x?x32x32xf16> -> memref<?x?x32x32xf16>
      %89 = loom.init_tensor %88[%4, %0, 32, 32] : memref<?x?x32x32xf16> -> tensor<?x?x32x32xf16>
      scf.if %80 {
        %90 = linalg.fill ins(%cst_1 : f16) outs(%28 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %91 = loom.sync ins(%74 : tensor<?x?x32x1xf16>) outs(%83 : tensor<?x?x32x1xf16>) -> tensor<?x?x32x1xf16>
        loom.semaphore_give %72 : memref<?x?x32x1xf16>
        %92 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%91 : tensor<?x?x32x1xf16>) outs(%90 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %105 = arith.maximumf %in, %out : f16
          linalg.yield %105 : f16
        } -> tensor<?x32x1xf16>
        %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%91, %92 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%83 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %105 = arith.subf %in, %in_3 : f16
          %106 = math.exp %105 : f16
          linalg.yield %106 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %27 : memref<?x32x1xf16>
        %94 = linalg.fill ins(%cst : f16) outs(%26 : tensor<?x32x1xf16>) -> tensor<?x32x1xf16>
        %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%93 : tensor<?x?x32x1xf16>) outs(%94 : tensor<?x32x1xf16>) {
        ^bb0(%in: f16, %out: f16):
          %105 = arith.addf %in, %out : f16
          linalg.yield %105 : f16
        } -> tensor<?x32x1xf16>
        %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%93, %95 : tensor<?x?x32x1xf16>, tensor<?x32x1xf16>) outs(%83 : tensor<?x?x32x1xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %105 = arith.divf %in, %in_3 : f16
          linalg.yield %105 : f16
        } -> tensor<?x?x32x1xf16>
        loom.semaphore_give %25 : memref<?x32x1xf16>
        %97 = loom.broadcast ins(%96 : tensor<?x?x32x1xf16>) outs(%89 : tensor<?x?x32x32xf16>) dim(3) -> tensor<?x?x32x128xf16>
        loom.semaphore_give %82 : memref<?x?x32x1xf16>
        %98 = loom.sync ins(%79 : tensor<?x?x32x128xf16>) outs(%86 : tensor<?x?x32x128xf16>) -> tensor<?x?x32x128xf16>
        loom.semaphore_give %77 : memref<?x?x32x128xf16>
        %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98, %97 : tensor<?x?x32x128xf16>, tensor<?x?x32x128xf16>) outs(%86 : tensor<?x?x32x128xf16>) {
        ^bb0(%in: f16, %in_3: f16, %out: f16):
          %105 = arith.mulf %in, %in_3 : f16
          linalg.yield %105 : f16
        } -> tensor<?x?x32x128xf16>
        loom.semaphore_give %88 : memref<?x?x32x32xf16>
        %100 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        %101 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%99 : tensor<?x?x32x128xf16>) outs(%100 : tensor<?x32x128xf16>) {
        ^bb0(%in: f16, %out: f16):
          %105 = arith.addf %in, %out : f16
          linalg.yield %105 : f16
        } -> tensor<?x32x128xf16>
        loom.semaphore_give %85 : memref<?x?x32x128xf16>
        %102 = loom.sync ins(%101 : tensor<?x32x128xf16>) outs(%18 : tensor<?x32x128xf16>) -> tensor<?x32x128xf16>
        loom.semaphore_give %7 : memref<?x32x128xf16>
        %103 = loom.subview %arg2[%5, 0, 0] [%0, 32, 128] [1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<16x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        %104 = loom.bufferize_to_memref %102 : tensor<?x32x128xf16> -> memref<?x32x128xf16>
        loom.copy %104, %103 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x32x128xf16> to memref<?x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.semaphore_give %17 : memref<?x32x128xf16>
      }
    }
    return
  }
}
