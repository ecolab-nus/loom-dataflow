module attributes {loom.tile_k = {is_reduction = false, upper_bound = 512 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
  func.func @matmul(%arg0: memref<4096x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x4096xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c4096 = arith.constant 4096 : index
    %c512 = arith.constant 512 : index
    %0 = loom.sym @tile_m {upper_bound = 4096 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 4096 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 512 : index} : index
    %3 = arith.ceildivui %c4096, %0 : index
    %4 = arith.ceildivui %c4096, %1 : index
    affine.parallel (%arg3, %arg4) = (0, 0) to (symbol(%3), symbol(%4)) {
      %5 = arith.ceildivui %c512, %2 : index
      %6 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %7 = loom.semaphore_take %6 : memref<?x?xf16> -> memref<?x?xf16>
      %8 = loom.init_tensor %7[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %9 = linalg.fill ins(%cst : f16) outs(%8 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %10 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %11 = loom.semaphore_take %10 : memref<?x?xf16> -> memref<?x?xf16>
      %12 = loom.init_tensor %11[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %13 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %14 = loom.semaphore_take %13 : memref<?x?xf16> -> memref<?x?xf16>
      %15 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %16 = loom.semaphore_take %15 : memref<?x?xf16> -> memref<?x?xf16>
      %17 = scf.for %arg5 = %c0 to %5 step %c1 iter_args(%arg6 = %9) -> (tensor<?x?xf16>) {
        %26 = arith.muli %arg3, %0 : index
        %27 = arith.muli %arg5, %2 : index
        %28 = loom.subview %arg0[%26, %27] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        loom.copy %28, %14 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
        %29 = loom.bufferize_to_tensor %14[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %30 = arith.muli %arg4, %1 : index
        %31 = loom.subview %arg1[%27, %30] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        loom.copy %31, %16 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
        %32 = loom.bufferize_to_tensor %16[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %33 = linalg.fill ins(%cst : f16) outs(%12 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %34 = linalg.matmul ins(%29, %32 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%33 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %16 : memref<?x?xf16>
        loom.semaphore_give %14 : memref<?x?xf16>
        %35 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg6, %34 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg6 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %36 = arith.addf %in, %in_0 : f16
          linalg.yield %36 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %11 : memref<?x?xf16>
        scf.yield %35 : tensor<?x?xf16>
      }
      %18 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
      %20 = loom.init_tensor %19[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %21 = linalg.copy ins(%17 : tensor<?x?xf16>) outs(%20 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %7 : memref<?x?xf16>
      %22 = arith.muli %arg3, %0 : index
      %23 = arith.muli %arg4, %1 : index
      %24 = loom.subview %arg2[%22, %23] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %25 = loom.bufferize_to_memref %21 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %25, %24 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %19 : memref<?x?xf16>
    }
    return
  }
}
