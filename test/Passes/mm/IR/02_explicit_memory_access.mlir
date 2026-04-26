module attributes {loom.tile_k = {is_reduction = false, upper_bound = 256 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @_matmul(%arg0: memref<4096x256xf16>, %arg1: memref<256x4096xf16>, %arg2: memref<4096x4096xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c4096 = arith.constant 4096 : index
    %c256 = arith.constant 256 : index
    %0 = loom.sym @tile_m {upper_bound = 4096 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 4096 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 256 : index} : index
    %3 = arith.ceildivui %c4096, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = arith.ceildivui %c256, %2 : index
      %6 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %7 = loom.semaphore_take %6 : memref<?x?xf16> -> memref<?x?xf16>
      %8 = loom.init_tensor %7[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %9 = loom.semaphore_take %6 : memref<?x?xf16> -> memref<?x?xf16>
      %10 = loom.init_tensor %9[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %11 = linalg.fill ins(%cst : f16) outs(%10 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %12 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %13 = loom.semaphore_take %12 : memref<?x?xf16> -> memref<?x?xf16>
      %14 = loom.init_tensor %13[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %15 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %16 = loom.semaphore_take %15 : memref<?x?xf16> -> memref<?x?xf16>
      %17 = loom.init_tensor %16[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %18 = loom.semaphore_take %15 : memref<?x?xf16> -> memref<?x?xf16>
      %19 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
      %21 = loom.init_tensor %20[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %22 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
      %23 = scf.for %arg5 = %c0 to %5 step %c1 iter_args(%arg6 = %11) -> (tensor<?x?xf16>) {
        %29 = arith.muli %arg3, %0 : index
        %30 = arith.muli %arg5, %2 : index
        %31 = loom.subview %arg0[%29, %30] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %31, %18 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %32 = loom.bufferize_to_tensor %18[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %33 = loom.sync ins(%32 : tensor<?x?xf16>) outs(%17 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %18 : memref<?x?xf16>
        %34 = arith.muli %arg4, %1 : index
        %35 = loom.subview %arg1[%30, %34] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        loom.copy %35, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
        %36 = loom.bufferize_to_tensor %22[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %37 = loom.sync ins(%36 : tensor<?x?xf16>) outs(%21 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %22 : memref<?x?xf16>
        %38 = linalg.fill ins(%cst : f16) outs(%14 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %39 = linalg.matmul ins(%33, %37 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%38 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %20 : memref<?x?xf16>
        loom.semaphore_give %16 : memref<?x?xf16>
        %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg6, %39 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg6 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %41 = arith.addf %in, %in_0 : f16
          linalg.yield %41 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %13 : memref<?x?xf16>
        scf.yield %40 : tensor<?x?xf16>
      }
      %24 = arith.muli %arg3, %0 : index
      %25 = arith.muli %arg4, %1 : index
      %26 = loom.sync ins(%23 : tensor<?x?xf16>) outs(%8 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %9 : memref<?x?xf16>
      %27 = loom.subview %arg2[%24, %25] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %28 = loom.bufferize_to_memref %26 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %28, %27 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %7 : memref<?x?xf16>
    }
    return
  }
}
