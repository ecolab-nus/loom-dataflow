module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x16xf16>, %arg5: memref<1x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_h {upper_bound = 16 : index} : index
    %4 = loom.sym @tile_c {upper_bound = 8 : index} : index
    %5 = arith.ceildivui %c16, %3 : index
    %6 = arith.ceildivui %c256, %0 : index
    %7 = arith.ceildivui %c64, %1 : index
    %8 = arith.ceildivui %c8, %4 : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (symbol(%5), symbol(%6), symbol(%7), 1, symbol(%8)) {
      %9 = arith.muli %arg8, %3 : index
      %10 = arith.muli %arg12, %4 : index
      %11 = arith.muli %arg9, %0 : index
      %12 = loom.alloc [1, 1, 1, %0] on @L1 : memref<1x1x1x?xf16>
      %13 = loom.semaphore_take %12 : memref<1x1x1x?xf16> -> memref<1x1x1x?xf16>
      %14 = loom.subview %arg1[%arg11, %9, %10, %11] [1, 1, 1, %0] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x16x8x256xf16> to memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>>
      loom.copy %14, %13 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>> to memref<1x1x1x?xf16>
      %15 = loom.bufferize_to_tensor %13[1, 1, 1, %0] : memref<1x1x1x?xf16> -> tensor<?xf16>
      %16 = loom.alloc [%0] on @L1 : memref<?xf32>
      %17 = loom.semaphore_take %16 : memref<?xf32> -> memref<?xf32>
      %18 = loom.init_tensor %17[%0] : memref<?xf32> -> tensor<?xf32>
      %19 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%15 : tensor<?xf16>) outs(%18 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %80 = arith.extf %in : f16 to f32
        linalg.yield %80 : f32
      } -> tensor<?xf32>
      loom.semaphore_give %13 : memref<1x1x1x?xf16>
      %20 = loom.alloc [%0] on @L1 : memref<?xf32>
      %21 = loom.semaphore_take %20 : memref<?xf32> -> memref<?xf32>
      %22 = loom.init_tensor %21[%0] : memref<?xf32> -> tensor<?xf32>
      %23 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%19 : tensor<?xf32>) outs(%22 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %80 = arith.truncf %cst_0 : f64 to f32
        %81 = arith.mulf %in, %80 : f32
        %82 = math.powf %cst, %81 : f32
        linalg.yield %82 : f32
      } -> tensor<?xf32>
      %24 = arith.muli %10, %c256 : index
      %25 = arith.divui %9, %c16 : index
      %26 = loom.alloc [1, %0, 1, 16] on @L1 : memref<1x?x1x16xf16>
      %27 = loom.semaphore_take %26 : memref<1x?x1x16xf16> -> memref<1x?x1x16xf16>
      %28 = loom.subview %arg4[%arg11, %24, %25, 0] [1, %0, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x1x16xf16> to memref<1x?x1x16xf16, strided<[32768, 16, 16, 1], offset: ?>>
      loom.copy %28, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x?x1x16xf16, strided<[32768, 16, 16, 1], offset: ?>> to memref<1x?x1x16xf16>
      %29 = loom.bufferize_to_tensor %27[1, %0, 1, 16] : memref<1x?x1x16xf16> -> tensor<?x16xf16>
      %30 = arith.muli %arg10, %1 : index
      %31 = loom.alloc [1, 1, 1, %1, 16] on @L1 : memref<1x1x1x?x16xf16>
      %32 = loom.semaphore_take %31 : memref<1x1x1x?x16xf16> -> memref<1x1x1x?x16xf16>
      %33 = loom.subview %arg5[%arg11, %10, %9, %30, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x8x16x64x16xf16> to memref<1x1x1x?x16xf16, strided<[131072, 16384, 1024, 16, 1], offset: ?>>
      loom.copy %33, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?x16xf16, strided<[131072, 16384, 1024, 16, 1], offset: ?>> to memref<1x1x1x?x16xf16>
      %34 = loom.bufferize_to_tensor %32[1, 1, 1, %1, 16] : memref<1x1x1x?x16xf16> -> tensor<?x16xf16>
      %35 = loom.alloc [16, %1] on @L1 : memref<16x?xf16>
      %36 = loom.semaphore_take %35 : memref<16x?xf16> -> memref<16x?xf16>
      %37 = loom.init_tensor %36[16, %1] : memref<16x?xf16> -> tensor<16x?xf16>
      %transposed = linalg.transpose ins(%34 : tensor<?x16xf16>) outs(%37 : tensor<16x?xf16>) permutation = [1, 0] 
      loom.semaphore_give %32 : memref<1x1x1x?x16xf16>
      %38 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
      %39 = loom.semaphore_take %38 : memref<?x?xf32> -> memref<?x?xf32>
      %40 = loom.init_tensor %39[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %41 = loom.semaphore_take %38 : memref<?x?xf32> -> memref<?x?xf32>
      %42 = loom.init_tensor %41[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %43 = linalg.fill ins(%cst_1 : f32) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %44 = linalg.matmul ins(%29, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%43 : tensor<?x?xf32>) -> tensor<?x?xf32>
      loom.semaphore_give %36 : memref<16x?xf16>
      loom.semaphore_give %27 : memref<1x?x1x16xf16>
      %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%44, %23 : tensor<?x?xf32>, tensor<?xf32>) outs(%42 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %80 = arith.mulf %in, %in_2 : f32
        linalg.yield %80 : f32
      } -> tensor<?x?xf32>
      loom.semaphore_give %39 : memref<?x?xf32>
      loom.semaphore_give %21 : memref<?xf32>
      %46 = arith.addi %arg9, %c1 : index
      %47 = arith.muli %46, %0 : index
      %48 = arith.ceildivui %47, %2 : index
      %49 = loom.alloc [1, 1, 1, %0, %2] on @L1 : memref<1x1x1x?x?xf16>
      %50 = loom.semaphore_take %49 : memref<1x1x1x?x?xf16> -> memref<1x1x1x?x?xf16>
      %51 = loom.alloc [1, 1, 1, %2] on @L1 : memref<1x1x1x?xf16>
      %52 = loom.semaphore_take %51 : memref<1x1x1x?xf16> -> memref<1x1x1x?xf16>
      %53 = loom.semaphore_take %51 : memref<1x1x1x?xf16> -> memref<1x1x1x?xf16>
      %54 = loom.alloc [%2] on @L1 : memref<?xf32>
      %55 = loom.semaphore_take %54 : memref<?xf32> -> memref<?xf32>
      %56 = loom.init_tensor %55[%2] : memref<?xf32> -> tensor<?xf32>
      %57 = loom.alloc [%2] on @L1 : memref<?xf32>
      %58 = loom.semaphore_take %57 : memref<?xf32> -> memref<?xf32>
      %59 = loom.init_tensor %58[%2] : memref<?xf32> -> tensor<?xf32>
      %60 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %61 = loom.semaphore_take %60 : memref<?x?xf16> -> memref<?x?xf16>
      %62 = loom.init_tensor %61[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %63 = loom.alloc [1, %2, 1, %1] on @L1 : memref<1x?x1x?xf16>
      %64 = loom.semaphore_take %63 : memref<1x?x1x?xf16> -> memref<1x?x1x?xf16>
      %65 = scf.for %arg13 = %c0 to %48 step %c1 iter_args(%arg14 = %45) -> (tensor<?x?xf32>) {
        %80 = arith.muli %arg13, %2 : index
        %81 = loom.subview %arg0[%arg11, %10, %25, %11, %80] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x8x1x256x256xf16> to memref<1x1x1x?x?xf16, strided<[524288, 65536, 65536, 256, 1], offset: ?>>
        loom.copy %81, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?x?xf16, strided<[524288, 65536, 65536, 256, 1], offset: ?>> to memref<1x1x1x?x?xf16>
        %82 = loom.bufferize_to_tensor %50[1, 1, 1, %0, %2] : memref<1x1x1x?x?xf16> -> tensor<?x?xf16>
        %83 = loom.subview %arg1[%arg11, %9, %10, %80] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x16x8x256xf16> to memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>>
        loom.copy %83, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>> to memref<1x1x1x?xf16>
        %84 = loom.bufferize_to_tensor %53[1, 1, 1, %2] : memref<1x1x1x?xf16> -> tensor<?xf16>
        %85 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%84 : tensor<?xf16>) outs(%56 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %93 = arith.extf %in : f16 to f32
          linalg.yield %93 : f32
        } -> tensor<?xf32>
        loom.semaphore_give %53 : memref<1x1x1x?xf16>
        %86 = loom.subview %arg2[%arg11, %9, %10, %80] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x16x8x256xf16> to memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>>
        loom.copy %86, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>> to memref<1x1x1x?xf16>
        %87 = loom.bufferize_to_tensor %52[1, 1, 1, %2] : memref<1x1x1x?xf16> -> tensor<?xf16>
        %88 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%87 : tensor<?xf16>) outs(%59 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %93 = arith.extf %in : f16 to f32
          linalg.yield %93 : f32
        } -> tensor<?xf32>
        loom.semaphore_give %52 : memref<1x1x1x?xf16>
        %89 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%82, %19, %85, %88 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%62 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
          %93 = arith.truncf %cst_0 : f64 to f32
          %94 = arith.mulf %in_3, %93 : f32
          %95 = arith.truncf %cst_0 : f64 to f32
          %96 = arith.mulf %in_2, %95 : f32
          %97 = arith.subf %96, %94 : f32
          %98 = math.powf %cst, %97 : f32
          %99 = arith.extf %in : f16 to f32
          %100 = arith.mulf %99, %98 : f32
          %101 = arith.mulf %100, %in_4 : f32
          %102 = arith.truncf %101 : f32 to f16
          linalg.yield %102 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %58 : memref<?xf32>
        loom.semaphore_give %55 : memref<?xf32>
        loom.semaphore_give %50 : memref<1x1x1x?x?xf16>
        %90 = loom.subview %arg3[%arg11, %24, %9, %30] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x16x64xf16> to memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>>
        loom.copy %90, %64 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>> to memref<1x?x1x?xf16>
        %91 = loom.bufferize_to_tensor %64[1, %2, 1, %1] : memref<1x?x1x?xf16> -> tensor<?x?xf16>
        %92 = linalg.matmul ins(%89, %91 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        loom.semaphore_give %64 : memref<1x?x1x?xf16>
        loom.semaphore_give %61 : memref<?x?xf16>
        scf.yield %92 : tensor<?x?xf32>
      }
      loom.semaphore_give %17 : memref<?xf32>
      %66 = loom.alloc [1] on @L1 : memref<1xf16>
      %67 = loom.semaphore_take %66 : memref<1xf16> -> memref<1xf16>
      %68 = loom.subview %arg6[%9] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<1xf16, strided<[1], offset: ?>>
      loom.copy %68, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1xf16, strided<[1], offset: ?>> to memref<1xf16>
      %69 = loom.bufferize_to_tensor %67[1] : memref<1xf16> -> tensor<f16>
      %70 = loom.alloc [1, %0, 1, %1] on @L1 : memref<1x?x1x?xf16>
      %71 = loom.semaphore_take %70 : memref<1x?x1x?xf16> -> memref<1x?x1x?xf16>
      %72 = loom.subview %arg3[%arg11, %24, %9, %30] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x16x64xf16> to memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>>
      loom.copy %72, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>> to memref<1x?x1x?xf16>
      %73 = loom.bufferize_to_tensor %71[1, %0, 1, %1] : memref<1x?x1x?xf16> -> tensor<?x?xf16>
      %74 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %75 = loom.semaphore_take %74 : memref<?x?xf16> -> memref<?x?xf16>
      %76 = loom.init_tensor %75[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %77 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %73, %69 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%76 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
        %80 = arith.extf %in_3 : f16 to f32
        %81 = arith.extf %in_2 : f16 to f32
        %82 = arith.mulf %81, %80 : f32
        %83 = arith.addf %in, %82 : f32
        %84 = arith.truncf %83 : f32 to f16
        linalg.yield %84 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %71 : memref<1x?x1x?xf16>
      loom.semaphore_give %67 : memref<1xf16>
      loom.semaphore_give %41 : memref<?x?xf32>
      %78 = loom.subview %arg7[%arg11, %24, %9, %30] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x16x64xf16> to memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>>
      %79 = loom.bufferize_to_memref %77 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %79, %78 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>>
      loom.semaphore_give %75 : memref<?x?xf16>
    }
    return
  }
}
