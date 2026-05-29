module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
  func.func @split_k_matmul_gather(%arg0: memref<512x4096xf16>, %arg1: memref<4096x512xf16>, %arg2: memref<512x512xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %c512 = arith.constant 512 : index
    %0 = loom.sym @tile_m {upper_bound = 512 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 512 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 4096 : index} : index
    %3 = arith.ceildivui %c512, %0 : index
    %4 = arith.ceildivui %c512, %1 : index
    %5 = arith.ceildivui %c4096, %2 : index
    affine.parallel (%arg3, %arg4, %arg5) = (0, 0, 0) to (symbol(%3), symbol(%4), symbol(%5)) {
      %6 = arith.muli %arg3, %0 : index
      %7 = arith.muli %arg5, %2 : index
      %8 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %9 = loom.semaphore_take %8 : memref<?x?xf16> -> memref<?x?xf16>
      %10 = loom.init_tensor %9[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %11 = loom.semaphore_take %8 : memref<?x?xf16> -> memref<?x?xf16>
      %12 = loom.subview %arg0[%6, %7] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.copy %12, %11 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
      %13 = loom.bufferize_to_tensor %11[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %14 = loom.sync ins(%13 : tensor<?x?xf16>) outs(%10 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %11 : memref<?x?xf16>
      %15 = arith.muli %arg5, %2 : index
      %16 = arith.muli %arg4, %1 : index
      %17 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %18 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
      %19 = loom.init_tensor %18[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %20 = loom.semaphore_take %17 : memref<?x?xf16> -> memref<?x?xf16>
      %21 = loom.subview %arg1[%15, %16] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
      loom.copy %21, %20 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
      %22 = loom.bufferize_to_tensor %20[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %23 = loom.sync ins(%22 : tensor<?x?xf16>) outs(%19 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %20 : memref<?x?xf16>
      %24 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
      %26 = loom.init_tensor %25[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %27 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
      %28 = loom.init_tensor %27[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %29 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
      %30 = loom.init_tensor %29[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %31 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
      %32 = loom.init_tensor %31[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %33 = linalg.fill ins(%cst : f16) outs(%32 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %34 = linalg.matmul ins(%14, %23 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %18 : memref<?x?xf16>
      loom.semaphore_give %9 : memref<?x?xf16>
      %35 = arith.ceildivui %c4096, %2 : index
      %36 = loom.sync ins(%34 : tensor<?x?xf16>) outs(%30 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %31 : memref<?x?xf16>
      %37 = loom.alloc [%35, %0, %1] on @L1 : memref<?x?x?xf16>
      %38 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
      %39 = loom.init_tensor %38[%35, %0, %1] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
      %40 = loom.semaphore_take %37 : memref<?x?x?xf16> -> memref<?x?x?xf16>
      %41 = loom.init_tensor %40[%35, %0, %1] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
      %42 = loom.gather ins(%36 : tensor<?x?xf16>) outs(%41 : tensor<?x?x?xf16>) across(%arg5 : index) -> tensor<?x?x?xf16>
      loom.semaphore_give %29 : memref<?x?xf16>
      %43 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %43 {
        %44 = linalg.fill ins(%cst : f16) outs(%28 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %45 = loom.sync ins(%42 : tensor<?x?x?xf16>) outs(%39 : tensor<?x?x?xf16>) -> tensor<?x?x?xf16>
        loom.semaphore_give %40 : memref<?x?x?xf16>
        %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%45 : tensor<?x?x?xf16>) outs(%44 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %52 = arith.addf %in, %out : f16
          linalg.yield %52 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %38 : memref<?x?x?xf16>
        %47 = arith.muli %arg3, %0 : index
        %48 = arith.muli %arg4, %1 : index
        %49 = loom.sync ins(%46 : tensor<?x?xf16>) outs(%26 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %27 : memref<?x?xf16>
        %50 = loom.subview %arg2[%47, %48] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        %51 = loom.bufferize_to_memref %49 : tensor<?x?xf16> -> memref<?x?xf16>
        loom.copy %51, %50 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        loom.semaphore_give %25 : memref<?x?xf16>
      }
    }
    return
  }
}
