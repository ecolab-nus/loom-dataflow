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
      %17 = loom.semaphore_take %15 : memref<?xf16> -> memref<?xf16>
      %18 = loom.init_tensor %17[%0] : memref<?xf16> -> tensor<?xf16>
      %19 = loom.subview %arg1[%11, %12, %13, %14] [1, 1, 1, %0] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      loom.copy %19, %16 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
      %20 = loom.bufferize_to_tensor %16[%0] : memref<?xf16> -> tensor<?xf16>
      %21 = loom.sync ins(%20 : tensor<?xf16>) outs(%18 : tensor<?xf16>) -> tensor<?xf16>
      loom.semaphore_give %16 : memref<?xf16>
      %22 = loom.alloc [%0, 32] on @L1 : memref<?x32xf16>
      %23 = loom.semaphore_take %22 : memref<?x32xf16> -> memref<?x32xf16>
      %24 = loom.init_tensor %23[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %25 = loom.semaphore_take %22 : memref<?x32xf16> -> memref<?x32xf16>
      %26 = loom.init_tensor %25[%0, 32] : memref<?x32xf16> -> tensor<?x32xf16>
      %27 = loom.broadcast ins(%21 : tensor<?xf16>) outs(%26 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
      %28 = arith.muli %13, %c256 : index
      %29 = arith.addi %14, %28 : index
      %30 = arith.divui %12, %c64 : index
      %31 = loom.alloc [%0, 64] on @L1 : memref<?x64xf16>
      %32 = loom.semaphore_take %31 : memref<?x64xf16> -> memref<?x64xf16>
      %33 = loom.init_tensor %32[%0, 64] : memref<?x64xf16> -> tensor<?x64xf16>
      %34 = loom.semaphore_take %31 : memref<?x64xf16> -> memref<?x64xf16>
      %35 = loom.subview %arg4[%11, %29, %30, 0] [1, %0, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      loom.copy %35, %34 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
      %36 = loom.bufferize_to_tensor %34[%0, 64] : memref<?x64xf16> -> tensor<?x64xf16>
      %37 = loom.sync ins(%36 : tensor<?x64xf16>) outs(%33 : tensor<?x64xf16>) -> tensor<?x64xf16>
      loom.semaphore_give %34 : memref<?x64xf16>
      %38 = arith.muli %arg10, %1 : index
      %39 = loom.alloc [64, %1] on @L1 : memref<64x?xf16>
      %40 = loom.semaphore_take %39 : memref<64x?xf16> -> memref<64x?xf16>
      %41 = loom.init_tensor %40[64, %1] : memref<64x?xf16> -> tensor<64x?xf16>
      %42 = loom.semaphore_take %39 : memref<64x?xf16> -> memref<64x?xf16>
      %43 = loom.subview %arg5[%11, %13, %12, 0, %38] [1, 1, 1, 64, %1] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
      loom.copy %43, %42 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
      %44 = loom.bufferize_to_tensor %42[64, %1] : memref<64x?xf16> -> tensor<64x?xf16>
      %45 = loom.sync ins(%44 : tensor<64x?xf16>) outs(%41 : tensor<64x?xf16>) -> tensor<64x?xf16>
      loom.semaphore_give %42 : memref<64x?xf16>
      %46 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %47 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
      %48 = loom.init_tensor %47[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %49 = loom.semaphore_take %46 : memref<?x?xf16> -> memref<?x?xf16>
      %50 = loom.init_tensor %49[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %51 = linalg.fill ins(%cst : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %52 = linalg.matmul ins(%37, %45 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%51 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %40 : memref<64x?xf16>
      loom.semaphore_give %32 : memref<?x64xf16>
      %53 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%52, %27 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%50 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %101 = math.exp %in_0 : f16
        %102 = arith.mulf %in, %101 : f16
        linalg.yield %102 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %47 : memref<?x?xf16>
      loom.semaphore_give %25 : memref<?x32xf16>
      %54 = arith.addi %arg9, %c1 : index
      %55 = arith.muli %54, %0 : index
      %56 = arith.ceildivui %55, %2 : index
      %57 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %58 = loom.semaphore_take %57 : memref<?x?xf16> -> memref<?x?xf16>
      %59 = loom.init_tensor %58[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %60 = loom.semaphore_take %57 : memref<?x?xf16> -> memref<?x?xf16>
      %61 = loom.init_tensor %60[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %62 = loom.semaphore_take %57 : memref<?x?xf16> -> memref<?x?xf16>
      %63 = loom.semaphore_take %57 : memref<?x?xf16> -> memref<?x?xf16>
      %64 = loom.init_tensor %63[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %65 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %66 = loom.semaphore_take %65 : memref<?x?xf16> -> memref<?x?xf16>
      %67 = loom.init_tensor %66[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %68 = loom.semaphore_take %65 : memref<?x?xf16> -> memref<?x?xf16>
      %69 = loom.alloc [%2] on @L1 : memref<?xf16>
      %70 = loom.semaphore_take %69 : memref<?xf16> -> memref<?xf16>
      %71 = loom.init_tensor %70[%2] : memref<?xf16> -> tensor<?xf16>
      %72 = loom.semaphore_take %69 : memref<?xf16> -> memref<?xf16>
      %73 = loom.semaphore_take %69 : memref<?xf16> -> memref<?xf16>
      %74 = loom.init_tensor %73[%2] : memref<?xf16> -> tensor<?xf16>
      %75 = loom.semaphore_take %69 : memref<?xf16> -> memref<?xf16>
      %76 = loom.alloc [32, %2] on @L1 : memref<32x?xf16>
      %77 = loom.semaphore_take %76 : memref<32x?xf16> -> memref<32x?xf16>
      %78 = loom.init_tensor %77[32, %2] : memref<32x?xf16> -> tensor<32x?xf16>
      %79 = loom.alloc [32, %2] on @L1 : memref<32x?xf16>
      %80 = loom.semaphore_take %79 : memref<32x?xf16> -> memref<32x?xf16>
      %81 = loom.init_tensor %80[32, %2] : memref<32x?xf16> -> tensor<32x?xf16>
      %82 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %83 = loom.semaphore_take %82 : memref<?x?xf16> -> memref<?x?xf16>
      %84 = loom.init_tensor %83[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %85 = loom.semaphore_take %82 : memref<?x?xf16> -> memref<?x?xf16>
      %86 = scf.for %arg13 = %c0 to %56 step %c1 iter_args(%arg14 = %53) -> (tensor<?x?xf16>) {
        %101 = arith.muli %arg13, %2 : index
        %102 = loom.subview %arg0[%11, %13, %30, %14, %101] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %102, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %103 = loom.bufferize_to_tensor %68[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %104 = loom.sync ins(%103 : tensor<?x?xf16>) outs(%67 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %68 : memref<?x?xf16>
        %105 = loom.subview %arg1[%11, %12, %13, %101] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %105, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %106 = loom.bufferize_to_tensor %75[%2] : memref<?xf16> -> tensor<?xf16>
        %107 = loom.sync ins(%106 : tensor<?xf16>) outs(%74 : tensor<?xf16>) -> tensor<?xf16>
        loom.semaphore_give %75 : memref<?xf16>
        %108 = loom.broadcast ins(%21 : tensor<?xf16>) outs(%24 : tensor<?x32xf16>) dim(1) -> tensor<?x?xf16>
        %109 = loom.broadcast ins(%107 : tensor<?xf16>) outs(%78 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        loom.semaphore_give %73 : memref<?xf16>
        %110 = loom.subview %arg2[%11, %12, %13, %101] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %110, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %111 = loom.bufferize_to_tensor %72[%2] : memref<?xf16> -> tensor<?xf16>
        %112 = loom.sync ins(%111 : tensor<?xf16>) outs(%71 : tensor<?xf16>) -> tensor<?xf16>
        loom.semaphore_give %72 : memref<?xf16>
        %113 = loom.broadcast ins(%112 : tensor<?xf16>) outs(%81 : tensor<32x?xf16>) dim(0) -> tensor<?x?xf16>
        loom.semaphore_give %70 : memref<?xf16>
        %114 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%104, %108, %109, %113 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%67 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_0: f16, %in_1: f16, %in_2: f16, %out: f16):
          %122 = arith.subf %in_0, %in_1 : f16
          %123 = math.exp %122 : f16
          %124 = arith.mulf %in, %123 : f16
          %125 = arith.mulf %124, %in_2 : f16
          linalg.yield %125 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %80 : memref<32x?xf16>
        loom.semaphore_give %77 : memref<32x?xf16>
        loom.semaphore_give %23 : memref<?x32xf16>
        %115 = arith.addi %101, %28 : index
        %116 = loom.subview %arg3[%11, %115, %12, %38] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        loom.copy %116, %85 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
        %117 = loom.bufferize_to_tensor %85[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %118 = loom.sync ins(%117 : tensor<?x?xf16>) outs(%84 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %85 : memref<?x?xf16>
        %119 = linalg.fill ins(%cst : f16) outs(%64 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %120 = linalg.matmul ins(%114, %118 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%119 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %83 : memref<?x?xf16>
        loom.semaphore_give %66 : memref<?x?xf16>
        %121 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %120 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %122 = arith.addf %in, %in_0 : f16
          linalg.yield %122 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %63 : memref<?x?xf16>
        scf.yield %121 : tensor<?x?xf16>
      }
      loom.semaphore_give %17 : memref<?xf16>
      %87 = loom.alloc [1] on @L1 : memref<f16>
      %88 = loom.semaphore_take %87 : memref<f16> -> memref<f16>
      %89 = loom.init_tensor %88[] : memref<f16> -> tensor<f16>
      %90 = loom.semaphore_take %87 : memref<f16> -> memref<f16>
      %91 = loom.subview %arg6[%12] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      loom.copy %91, %90 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
      %92 = loom.bufferize_to_tensor %90[] : memref<f16> -> tensor<f16>
      %93 = loom.sync ins(%92 : tensor<f16>) outs(%89 : tensor<f16>) -> tensor<f16>
      loom.semaphore_give %90 : memref<f16>
      %94 = loom.subview %arg3[%11, %29, %12, %38] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.copy %94, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
      %95 = loom.bufferize_to_tensor %62[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %96 = loom.sync ins(%95 : tensor<?x?xf16>) outs(%61 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %62 : memref<?x?xf16>
      %97 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%86, %96, %93 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%61 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
        %101 = arith.mulf %in_0, %in_1 : f16
        %102 = arith.addf %in, %101 : f16
        linalg.yield %102 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %88 : memref<f16>
      loom.semaphore_give %49 : memref<?x?xf16>
      %98 = loom.sync ins(%97 : tensor<?x?xf16>) outs(%59 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %60 : memref<?x?xf16>
      %99 = loom.subview %arg7[%11, %29, %12, %38] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %100 = loom.bufferize_to_memref %98 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %100, %99 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %58 : memref<?x?xf16>
    }
    return
  }
}
