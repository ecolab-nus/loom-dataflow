module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_h {upper_bound = 64 : index} : index
    %4 = loom.sym @tile_b {upper_bound = 2 : index} : index
    %5 = loom.sym @tile_c {upper_bound = 8 : index} : index
    %6 = arith.ceildivui %c64, %3 : index
    %7 = arith.ceildivui %c256, %0 : index
    %8 = arith.ceildivui %c64, %1 : index
    %9 = arith.ceildivui %c2, %4 : index
    %10 = arith.ceildivui %c8, %5 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%6), symbol(%7), symbol(%8), symbol(%9), symbol(%10)) {
      %11 = arith.muli %arg11, %4 : index
      %12 = arith.muli %arg8, %3 : index
      %13 = arith.muli %arg12, %5 : index
      %14 = arith.muli %arg9, %0 : index
      %15 = loom.alloc [%0] on @L1 : memref<?xf16>
      %16 = loom.semaphore_take %15 : memref<?xf16> -> memref<?xf16>
      %17 = loom.subview %arg1[%11, %12, %13, %14] [1, 1, 1, %0] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      loom.copy %17, %16 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
      %18 = loom.bufferize_to_tensor %16[%0] : memref<?xf16> -> tensor<?xf16>
      %19 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %20 = loom.semaphore_take %19 : memref<?x32xf16> -> memref<?x32xf16>
      %21 = loom.init_tensor %20[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %22 = loom.semaphore_take %19 : memref<?x32xf16> -> memref<?x32xf16>
      %23 = loom.init_tensor %22[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %24 = loom.broadcast ins(%18 : tensor<?xf16>) outs(%23 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
      %25 = arith.muli %13, %c256 : index
      %26 = arith.addi %14, %25 : index
      %27 = arith.divui %12, %c64 : index
      %28 = loom.alloc [%0, 64] on @L1 : memref<?x64xf16>
      %29 = loom.semaphore_take %28 : memref<?x64xf16> -> memref<?x64xf16>
      %30 = loom.subview %arg4[%11, %26, %27, 0] [1, %0, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      loom.copy %30, %29 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
      %31 = loom.bufferize_to_tensor %29[%0, 64] : memref<?x64xf16> -> tensor<?x64xf16>
      %32 = arith.muli %arg10, %1 : index
      %33 = loom.alloc [64, %1] on @L1 : memref<64x?xf16>
      %34 = loom.semaphore_take %33 : memref<64x?xf16> -> memref<64x?xf16>
      %35 = loom.subview %arg5[%11, %13, %12, 0, %32] [1, 1, 1, 64, %1] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
      loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
      %36 = loom.bufferize_to_tensor %34[64, %1] : memref<64x?xf16> -> tensor<64x?xf16>
      %37 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
      %39 = loom.init_tensor %38[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %40 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
      %41 = loom.init_tensor %40[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %42 = linalg.fill ins(%cst : f16) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %43 = linalg.matmul ins(%31, %36 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %34 : memref<64x?xf16>
      loom.semaphore_give %29 : memref<?x64xf16>
      %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%43, %24 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%41 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %83 = math.exp %in_0 : f16
        %84 = arith.mulf %in, %83 : f16
        linalg.yield %84 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %38 : memref<?x?xf16>
      loom.semaphore_give %22 : memref<?x32xf16>
      %45 = arith.addi %arg9, %c1 : index
      %46 = arith.muli %45, %0 : index
      %47 = arith.ceildivui %46, %2 : index
      %48 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %49 = loom.semaphore_take %48 : memref<?x?xf16> -> memref<?x?xf16>
      %50 = loom.init_tensor %49[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %51 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %52 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
      %53 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
      %55 = loom.init_tensor %54[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %56 = loom.alloc [%2] on @L1 : memref<?xf16>
      %57 = loom.semaphore_take %56 : memref<?xf16> -> memref<?xf16>
      %58 = loom.alloc [%2] on @L1 : memref<?xf16>
      %59 = loom.semaphore_take %58 : memref<?xf16> -> memref<?xf16>
      %60 = loom.alloc [32, %2] on @L1 : memref<32x?xf16>
      %61 = loom.semaphore_take %60 : memref<32x?xf16> -> memref<32x?xf16>
      %62 = loom.init_tensor %61[32, %2] : memref<32x?xf16> -> tensor<32x?xf16>
      %63 = loom.alloc [32, %2] on @L1 : memref<32x?xf16>
      %64 = loom.semaphore_take %63 : memref<32x?xf16> -> memref<32x?xf16>
      %65 = loom.init_tensor %64[32, %2] : memref<32x?xf16> -> tensor<32x?xf16>
      %66 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %67 = loom.semaphore_take %66 : memref<?x?xf16> -> memref<?x?xf16>
      %68 = scf.for %arg13 = %c0 to %47 step %c1 iter_args(%arg14 = %44) -> (tensor<?x?xf16>) {
        %83 = arith.muli %arg13, %2 : index
        %84 = loom.subview %arg0[%11, %13, %27, %14, %83] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %84, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %85 = loom.bufferize_to_tensor %52[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %86 = loom.subview %arg1[%11, %12, %13, %83] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %86, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %87 = loom.bufferize_to_tensor %57[%2] : memref<?xf16> -> tensor<?xf16>
        %88 = loom.broadcast ins(%18 : tensor<?xf16>) outs(%21 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
        %89 = loom.broadcast ins(%87 : tensor<?xf16>) outs(%62 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        loom.semaphore_give %57 : memref<?xf16>
        %90 = loom.subview %arg2[%11, %12, %13, %83] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %90, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %91 = loom.bufferize_to_tensor %59[%2] : memref<?xf16> -> tensor<?xf16>
        %92 = loom.broadcast ins(%91 : tensor<?xf16>) outs(%65 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        loom.semaphore_give %59 : memref<?xf16>
        %93 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %88, %89, %92 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
          %100 = arith.subf %in_0, %in_1 : f16
          %101 = math.exp %100 : f16
          %102 = arith.mulf %in, %101 : f16
          %103 = arith.mulf %102, %in_2 : f16
          linalg.yield %103 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %64 : memref<32x?xf16>
        loom.semaphore_give %61 : memref<32x?xf16>
        loom.semaphore_give %52 : memref<?x?xf16>
        loom.semaphore_give %20 : memref<?x32xf16>
        %94 = arith.addi %83, %25 : index
        %95 = loom.subview %arg3[%11, %94, %12, %32] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        loom.copy %95, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
        %96 = loom.bufferize_to_tensor %67[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %97 = linalg.fill ins(%cst : f16) outs(%50 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %98 = linalg.matmul ins(%93, %96 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%97 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %67 : memref<?x?xf16>
        loom.semaphore_give %54 : memref<?x?xf16>
        %99 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %98 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %100 = arith.addf %in, %in_0 : f16
          linalg.yield %100 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %49 : memref<?x?xf16>
        scf.yield %99 : tensor<?x?xf16>
      }
      loom.semaphore_give %16 : memref<?xf16>
      %69 = loom.alloc [1] on @L1 : memref<f16>
      %70 = loom.semaphore_take %69 : memref<f16> -> memref<f16>
      %71 = loom.subview %arg6[%12] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      loom.copy %71, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
      %72 = loom.bufferize_to_tensor %70[] : memref<f16> -> tensor<f16>
      %73 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %74 = loom.semaphore_take %73 : memref<?x?xf16> -> memref<?x?xf16>
      %75 = loom.subview %arg3[%11, %26, %12, %32] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.copy %75, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
      %76 = loom.bufferize_to_tensor %74[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %77 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %78 = loom.semaphore_take %77 : memref<?x?xf16> -> memref<?x?xf16>
      %79 = loom.init_tensor %78[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%68, %76, %72 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%79 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
        %83 = arith.mulf %in_0, %in_1 : f16
        %84 = arith.addf %in, %83 : f16
        linalg.yield %84 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %70 : memref<f16>
      loom.semaphore_give %74 : memref<?x?xf16>
      loom.semaphore_give %40 : memref<?x?xf16>
      %81 = loom.subview %arg7[%11, %26, %12, %32] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %82 = loom.bufferize_to_memref %80 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %82, %81 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %78 : memref<?x?xf16>
    }
    return
  }
}
