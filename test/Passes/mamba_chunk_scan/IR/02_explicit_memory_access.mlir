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
      %42 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
      %43 = loom.init_tensor %42[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %44 = linalg.fill ins(%cst : f16) outs(%41 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %45 = linalg.matmul ins(%31, %36 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %34 : memref<64x?xf16>
      loom.semaphore_give %29 : memref<?x64xf16>
      %46 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%45, %24 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%43 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %78 = math.exp %in_0 : f16
        %79 = arith.mulf %in, %78 : f16
        linalg.yield %79 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %40 : memref<?x?xf16>
      loom.semaphore_give %22 : memref<?x32xf16>
      %47 = arith.addi %arg9, %c1 : index
      %48 = arith.muli %47, %0 : index
      %49 = arith.ceildivui %48, %2 : index
      %50 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %51 = loom.semaphore_take %50 : memref<?x?xf16> -> memref<?x?xf16>
      %52 = loom.init_tensor %51[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %53 = loom.alloc [%2] on @L1 : memref<?xf16>
      %54 = loom.semaphore_take %53 : memref<?xf16> -> memref<?xf16>
      %55 = loom.semaphore_take %53 : memref<?xf16> -> memref<?xf16>
      %56 = loom.alloc [32, %2] on @L1 : memref<32x?xf16>
      %57 = loom.semaphore_take %56 : memref<32x?xf16> -> memref<32x?xf16>
      %58 = loom.init_tensor %57[32, %2] : memref<32x?xf16> -> tensor<32x?xf16>
      %59 = loom.alloc [32, %2] on @L1 : memref<32x?xf16>
      %60 = loom.semaphore_take %59 : memref<32x?xf16> -> memref<32x?xf16>
      %61 = loom.init_tensor %60[32, %2] : memref<32x?xf16> -> tensor<32x?xf16>
      %62 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %63 = loom.semaphore_take %62 : memref<?x?xf16> -> memref<?x?xf16>
      %64 = scf.for %arg13 = %c0 to %49 step %c1 iter_args(%arg14 = %46) -> (tensor<?x?xf16>) {
        %78 = arith.muli %arg13, %2 : index
        %79 = loom.subview %arg0[%11, %13, %27, %14, %78] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %79, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %80 = loom.bufferize_to_tensor %51[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %81 = loom.subview %arg1[%11, %12, %13, %78] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %81, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %82 = loom.bufferize_to_tensor %55[%2] : memref<?xf16> -> tensor<?xf16>
        %83 = loom.broadcast ins(%18 : tensor<?xf16>) outs(%21 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
        %84 = loom.broadcast ins(%82 : tensor<?xf16>) outs(%58 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        loom.semaphore_give %55 : memref<?xf16>
        %85 = loom.subview %arg2[%11, %12, %13, %78] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %85, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %86 = loom.bufferize_to_tensor %54[%2] : memref<?xf16> -> tensor<?xf16>
        %87 = loom.broadcast ins(%86 : tensor<?xf16>) outs(%61 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        loom.semaphore_give %54 : memref<?xf16>
        %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80, %83, %84, %87 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%52 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
          %93 = arith.subf %in_0, %in_1 : f16
          %94 = math.exp %93 : f16
          %95 = arith.mulf %in, %94 : f16
          %96 = arith.mulf %95, %in_2 : f16
          linalg.yield %96 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %60 : memref<32x?xf16>
        loom.semaphore_give %57 : memref<32x?xf16>
        loom.semaphore_give %20 : memref<?x32xf16>
        %89 = arith.addi %78, %25 : index
        %90 = loom.subview %arg3[%11, %89, %12, %32] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        loom.copy %90, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
        %91 = loom.bufferize_to_tensor %63[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %92 = linalg.matmul ins(%88, %91 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %63 : memref<?x?xf16>
        loom.semaphore_give %51 : memref<?x?xf16>
        scf.yield %92 : tensor<?x?xf16>
      }
      loom.semaphore_give %16 : memref<?xf16>
      %65 = loom.alloc [1] on @L1 : memref<f16>
      %66 = loom.semaphore_take %65 : memref<f16> -> memref<f16>
      %67 = loom.subview %arg6[%12] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      loom.copy %67, %66 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
      %68 = loom.bufferize_to_tensor %66[] : memref<f16> -> tensor<f16>
      %69 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %70 = loom.semaphore_take %69 : memref<?x?xf16> -> memref<?x?xf16>
      %71 = loom.init_tensor %70[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %72 = loom.subview %arg3[%11, %26, %12, %32] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.copy %72, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
      %73 = loom.bufferize_to_tensor %70[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %74 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%64, %73, %68 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%71 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
        %78 = arith.mulf %in_0, %in_1 : f16
        %79 = arith.addf %in, %78 : f16
        linalg.yield %79 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %66 : memref<f16>
      loom.semaphore_give %42 : memref<?x?xf16>
      %75 = loom.sync ins(%74 : tensor<?x?xf16>) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %70 : memref<?x?xf16>
      %76 = loom.subview %arg7[%11, %26, %12, %32] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %77 = loom.bufferize_to_memref %75 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %77, %76 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %38 : memref<?x?xf16>
    }
    return
  }
}
