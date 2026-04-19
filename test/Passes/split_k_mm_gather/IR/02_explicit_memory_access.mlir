module attributes {loom.tile_k = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_m = {is_reduction = false, upper_bound = 512 : index}, loom.tile_n = {is_reduction = false, upper_bound = 512 : index}} {
  func.func @split_k_matmul_gather(%arg0: memref<512x512xf16>, %arg1: memref<512x4096xf16>, %arg2: memref<4096x512xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f16
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
      %10 = loom.subview %arg1[%6, %7] [%0, %2] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x4096xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.copy %10, %9 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
      %11 = loom.bufferize_to_tensor %9[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %12 = arith.muli %arg5, %2 : index
      %13 = arith.muli %arg4, %1 : index
      %14 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %15 = loom.semaphore_take %14 : memref<?x?xf16> -> memref<?x?xf16>
      %16 = loom.subview %arg2[%12, %13] [%2, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<4096x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
      loom.copy %16, %15 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>> to memref<?x?xf16>
      %17 = loom.bufferize_to_tensor %15[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %18 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %19 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
      %20 = loom.init_tensor %19[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %21 = loom.semaphore_take %18 : memref<?x?xf16> -> memref<?x?xf16>
      %22 = loom.init_tensor %21[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %23 = linalg.fill ins(%cst : f16) outs(%22 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %24 = linalg.matmul ins(%11, %17 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %15 : memref<?x?xf16>
      loom.semaphore_give %9 : memref<?x?xf16>
      %25 = arith.ceildivui %c4096, %2 : index
      %26 = loom.alloc [%25, %0, %1] on @L1 : memref<?x?x?xf16>
      %27 = loom.semaphore_take %26 : memref<?x?x?xf16> -> memref<?x?x?xf16>
      %28 = loom.init_tensor %27[%25, %0, %1] : memref<?x?x?xf16> -> tensor<?x?x?xf16>
      %29 = loom.gather ins(%24 : tensor<?x?xf16>) outs(%28 : tensor<?x?x?xf16>) across(%arg5 : index) -> tensor<?x?x?xf16>
      loom.semaphore_give %21 : memref<?x?xf16>
      %30 = arith.cmpi eq, %arg5, %c0 : index
      scf.if %30 {
        %31 = linalg.fill ins(%cst : f16) outs(%20 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %32 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%29 : tensor<?x?x?xf16>) outs(%31 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %out: f16):
          %37 = arith.addf %in, %out : f16
          linalg.yield %37 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %27 : memref<?x?x?xf16>
        %33 = arith.muli %arg3, %0 : index
        %34 = arith.muli %arg4, %1 : index
        %35 = loom.subview %arg0[%33, %34] [%0, %1] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        %36 = loom.bufferize_to_memref %32 : tensor<?x?xf16> -> memref<?x?xf16>
        loom.copy %36, %35 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[512, 1], offset: ?>>
        loom.semaphore_give %19 : memref<?x?xf16>
      }
    }
    return
  }
}
