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
      %12 = loom.alloc [%0] on @L1 : memref<?xf16>
      %13 = loom.semaphore_take %12 : memref<?xf16> -> memref<?xf16>
      %14 = loom.subview %arg1[%arg11, %9, %10, %11] [1, 1, 1, %0] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      loom.copy %14, %13 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
      %15 = loom.bufferize_to_tensor %13[%0] : memref<?xf16> -> tensor<?xf16>
      %16 = loom.alloc [%0] on @L1 : memref<?xf32>
      %17 = loom.semaphore_take %16 : memref<?xf32> -> memref<?xf32>
      %18 = loom.init_tensor %17[%0] : memref<?xf32> -> tensor<?xf32>
      %19 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%15 : tensor<?xf16>) outs(%18 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %76 = arith.extf %in : f16 to f32
        linalg.yield %76 : f32
      } -> tensor<?xf32>
      loom.semaphore_give %13 : memref<?xf16>
      %20 = loom.alloc [%0] on @L1 : memref<?xf32>
      %21 = loom.semaphore_take %20 : memref<?xf32> -> memref<?xf32>
      %22 = loom.init_tensor %21[%0] : memref<?xf32> -> tensor<?xf32>
      %23 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%19 : tensor<?xf32>) outs(%22 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %76 = arith.truncf %cst_0 : f64 to f32
        %77 = arith.mulf %in, %76 : f32
        %78 = math.powf %cst, %77 : f32
        linalg.yield %78 : f32
      } -> tensor<?xf32>
      %24 = arith.muli %10, %c256 : index
      %25 = arith.divui %9, %c16 : index
      %26 = loom.alloc [%0, 16] on @L1 : memref<?x16xf16>
      %27 = loom.semaphore_take %26 : memref<?x16xf16> -> memref<?x16xf16>
      %28 = loom.subview %arg4[%arg11, %24, %25, 0] [1, %0, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      loom.copy %28, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
      %29 = loom.bufferize_to_tensor %27[%0, 16] : memref<?x16xf16> -> tensor<?x16xf16>
      %30 = arith.muli %arg10, %1 : index
      %31 = loom.alloc [%1, 16] on @L1 : memref<?x16xf16>
      %32 = loom.semaphore_take %31 : memref<?x16xf16> -> memref<?x16xf16>
      %33 = loom.subview %arg5[%arg11, %10, %9, %30, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      loom.copy %33, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
      %34 = loom.bufferize_to_tensor %32[%1, 16] : memref<?x16xf16> -> tensor<?x16xf16>
      %35 = loom.alloc [16, %1] on @L1 : memref<16x?xf16>
      %36 = loom.semaphore_take %35 : memref<16x?xf16> -> memref<16x?xf16>
      %37 = loom.init_tensor %36[16, %1] : memref<16x?xf16> -> tensor<16x?xf16>
      %transposed = linalg.transpose ins(%34 : tensor<?x16xf16>) outs(%37 : tensor<16x?xf16>) permutation = [1, 0] 
      loom.semaphore_give %32 : memref<?x16xf16>
      %38 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
      %39 = loom.semaphore_take %38 : memref<?x?xf32> -> memref<?x?xf32>
      %40 = loom.init_tensor %39[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %41 = loom.semaphore_take %38 : memref<?x?xf32> -> memref<?x?xf32>
      %42 = loom.init_tensor %41[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %43 = linalg.fill ins(%cst_1 : f32) outs(%40 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %44 = linalg.matmul ins(%29, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%43 : tensor<?x?xf32>) -> tensor<?x?xf32>
      loom.semaphore_give %36 : memref<16x?xf16>
      loom.semaphore_give %27 : memref<?x16xf16>
      %45 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%44, %23 : tensor<?x?xf32>, tensor<?xf32>) outs(%42 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %76 = arith.mulf %in, %in_2 : f32
        linalg.yield %76 : f32
      } -> tensor<?x?xf32>
      loom.semaphore_give %39 : memref<?x?xf32>
      loom.semaphore_give %21 : memref<?xf32>
      %46 = arith.addi %arg9, %c1 : index
      %47 = arith.muli %46, %0 : index
      %48 = arith.ceildivui %47, %2 : index
      %49 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
      %51 = loom.init_tensor %50[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %52 = loom.alloc [%2] on @L1 : memref<?xf16>
      %53 = loom.semaphore_take %52 : memref<?xf16> -> memref<?xf16>
      %54 = loom.semaphore_take %52 : memref<?xf16> -> memref<?xf16>
      %55 = loom.alloc [%2] on @L1 : memref<?xf32>
      %56 = loom.semaphore_take %55 : memref<?xf32> -> memref<?xf32>
      %57 = loom.init_tensor %56[%2] : memref<?xf32> -> tensor<?xf32>
      %58 = loom.alloc [%2] on @L1 : memref<?xf32>
      %59 = loom.semaphore_take %58 : memref<?xf32> -> memref<?xf32>
      %60 = loom.init_tensor %59[%2] : memref<?xf32> -> tensor<?xf32>
      %61 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %62 = loom.semaphore_take %61 : memref<?x?xf16> -> memref<?x?xf16>
      %63 = scf.for %arg13 = %c0 to %48 step %c1 iter_args(%arg14 = %45) -> (tensor<?x?xf32>) {
        %76 = arith.muli %arg13, %2 : index
        %77 = loom.subview %arg0[%arg11, %10, %25, %11, %76] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %77, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %78 = loom.bufferize_to_tensor %50[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %79 = loom.subview %arg1[%arg11, %9, %10, %76] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %79, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %80 = loom.bufferize_to_tensor %54[%2] : memref<?xf16> -> tensor<?xf16>
        %81 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%80 : tensor<?xf16>) outs(%57 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %89 = arith.extf %in : f16 to f32
          linalg.yield %89 : f32
        } -> tensor<?xf32>
        loom.semaphore_give %54 : memref<?xf16>
        %82 = loom.subview %arg2[%arg11, %9, %10, %76] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %82, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %83 = loom.bufferize_to_tensor %53[%2] : memref<?xf16> -> tensor<?xf16>
        %84 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%83 : tensor<?xf16>) outs(%60 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %89 = arith.extf %in : f16 to f32
          linalg.yield %89 : f32
        } -> tensor<?xf32>
        loom.semaphore_give %53 : memref<?xf16>
        %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%78, %19, %81, %84 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%51 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
          %89 = arith.truncf %cst_0 : f64 to f32
          %90 = arith.mulf %in_3, %89 : f32
          %91 = arith.truncf %cst_0 : f64 to f32
          %92 = arith.mulf %in_2, %91 : f32
          %93 = arith.subf %92, %90 : f32
          %94 = math.powf %cst, %93 : f32
          %95 = arith.extf %in : f16 to f32
          %96 = arith.mulf %95, %94 : f32
          %97 = arith.mulf %96, %in_4 : f32
          %98 = arith.truncf %97 : f32 to f16
          linalg.yield %98 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %59 : memref<?xf32>
        loom.semaphore_give %56 : memref<?xf32>
        %86 = loom.subview %arg3[%arg11, %24, %9, %30] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        loom.copy %86, %62 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
        %87 = loom.bufferize_to_tensor %62[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %88 = linalg.matmul ins(%85, %87 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        loom.semaphore_give %62 : memref<?x?xf16>
        loom.semaphore_give %50 : memref<?x?xf16>
        scf.yield %88 : tensor<?x?xf32>
      }
      loom.semaphore_give %17 : memref<?xf32>
      %64 = loom.alloc [1] on @L1 : memref<f16>
      %65 = loom.semaphore_take %64 : memref<f16> -> memref<f16>
      %66 = loom.subview %arg6[%9] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      loom.copy %66, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
      %67 = loom.bufferize_to_tensor %65[] : memref<f16> -> tensor<f16>
      %68 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %69 = loom.semaphore_take %68 : memref<?x?xf16> -> memref<?x?xf16>
      %70 = loom.init_tensor %69[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %71 = loom.subview %arg3[%arg11, %24, %9, %30] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      loom.copy %71, %69 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
      %72 = loom.bufferize_to_tensor %69[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%63, %72, %67 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%70 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
        %76 = arith.extf %in_3 : f16 to f32
        %77 = arith.extf %in_2 : f16 to f32
        %78 = arith.mulf %77, %76 : f32
        %79 = arith.addf %in, %78 : f32
        %80 = arith.truncf %79 : f32 to f16
        linalg.yield %80 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %65 : memref<f16>
      loom.semaphore_give %41 : memref<?x?xf32>
      %74 = loom.subview %arg7[%arg11, %24, %9, %30] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      loom.semaphore_give %69 : memref<?x?xf16>
    }
    return
  }
}
