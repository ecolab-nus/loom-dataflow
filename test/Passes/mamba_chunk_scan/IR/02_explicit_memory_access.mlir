module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_h {upper_bound = 16 : index} : index
    %4 = loom.sym @tile_b {upper_bound = 2 : index} : index
    %5 = loom.sym @tile_c {upper_bound = 8 : index} : index
    %6 = arith.ceildivui %c16, %3 : index
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
      %17 = loom.subview %arg1[%11, %12, %13, %14] [1, 1, 1, %0] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      loom.copy %17, %16 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
      %18 = loom.bufferize_to_tensor %16[%0] : memref<?xf16> -> tensor<?xf16>
      %19 = loom.alloc [%0] on @L1 : memref<?xf32>
      %20 = loom.semaphore_take %19 : memref<?xf32> -> memref<?xf32>
      %21 = loom.init_tensor %20[%0] : memref<?xf32> -> tensor<?xf32>
      %22 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%18 : tensor<?xf16>) outs(%21 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %79 = arith.extf %in : f16 to f32
        linalg.yield %79 : f32
      } -> tensor<?xf32>
      loom.semaphore_give %16 : memref<?xf16>
      %23 = loom.alloc [%0] on @L1 : memref<?xf32>
      %24 = loom.semaphore_take %23 : memref<?xf32> -> memref<?xf32>
      %25 = loom.init_tensor %24[%0] : memref<?xf32> -> tensor<?xf32>
      %26 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%22 : tensor<?xf32>) outs(%25 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %79 = arith.truncf %cst_0 : f64 to f32
        %80 = arith.mulf %in, %79 : f32
        %81 = math.powf %cst, %80 : f32
        linalg.yield %81 : f32
      } -> tensor<?xf32>
      %27 = arith.muli %13, %c256 : index
      %28 = arith.divui %12, %c16 : index
      %29 = loom.alloc [%0, 16] on @L1 : memref<?x16xf16>
      %30 = loom.semaphore_take %29 : memref<?x16xf16> -> memref<?x16xf16>
      %31 = loom.subview %arg4[%11, %27, %28, 0] [1, %0, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      loom.copy %31, %30 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
      %32 = loom.bufferize_to_tensor %30[%0, 16] : memref<?x16xf16> -> tensor<?x16xf16>
      %33 = arith.muli %arg10, %1 : index
      %34 = loom.alloc [%1, 16] on @L1 : memref<?x16xf16>
      %35 = loom.semaphore_take %34 : memref<?x16xf16> -> memref<?x16xf16>
      %36 = loom.subview %arg5[%11, %13, %12, %33, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      loom.copy %36, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
      %37 = loom.bufferize_to_tensor %35[%1, 16] : memref<?x16xf16> -> tensor<?x16xf16>
      %38 = loom.alloc [16, %1] on @L1 : memref<16x?xf16>
      %39 = loom.semaphore_take %38 : memref<16x?xf16> -> memref<16x?xf16>
      %40 = loom.init_tensor %39[16, %1] : memref<16x?xf16> -> tensor<16x?xf16>
      %transposed = linalg.transpose ins(%37 : tensor<?x16xf16>) outs(%40 : tensor<16x?xf16>) permutation = [1, 0] 
      loom.semaphore_give %35 : memref<?x16xf16>
      %41 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
      %42 = loom.semaphore_take %41 : memref<?x?xf32> -> memref<?x?xf32>
      %43 = loom.init_tensor %42[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %44 = loom.semaphore_take %41 : memref<?x?xf32> -> memref<?x?xf32>
      %45 = loom.init_tensor %44[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %46 = linalg.fill ins(%cst_1 : f32) outs(%43 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %47 = linalg.matmul ins(%32, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%46 : tensor<?x?xf32>) -> tensor<?x?xf32>
      loom.semaphore_give %39 : memref<16x?xf16>
      loom.semaphore_give %30 : memref<?x16xf16>
      %48 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%47, %26 : tensor<?x?xf32>, tensor<?xf32>) outs(%45 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %79 = arith.mulf %in, %in_2 : f32
        linalg.yield %79 : f32
      } -> tensor<?x?xf32>
      loom.semaphore_give %42 : memref<?x?xf32>
      loom.semaphore_give %24 : memref<?xf32>
      %49 = arith.addi %arg9, %c1 : index
      %50 = arith.muli %49, %0 : index
      %51 = arith.ceildivui %50, %2 : index
      %52 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %53 = loom.semaphore_take %52 : memref<?x?xf16> -> memref<?x?xf16>
      %54 = loom.init_tensor %53[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %55 = loom.alloc [%2] on @L1 : memref<?xf16>
      %56 = loom.semaphore_take %55 : memref<?xf16> -> memref<?xf16>
      %57 = loom.semaphore_take %55 : memref<?xf16> -> memref<?xf16>
      %58 = loom.alloc [%2] on @L1 : memref<?xf32>
      %59 = loom.semaphore_take %58 : memref<?xf32> -> memref<?xf32>
      %60 = loom.init_tensor %59[%2] : memref<?xf32> -> tensor<?xf32>
      %61 = loom.alloc [%2] on @L1 : memref<?xf32>
      %62 = loom.semaphore_take %61 : memref<?xf32> -> memref<?xf32>
      %63 = loom.init_tensor %62[%2] : memref<?xf32> -> tensor<?xf32>
      %64 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %65 = loom.semaphore_take %64 : memref<?x?xf16> -> memref<?x?xf16>
      %66 = scf.for %arg13 = %c0 to %51 step %c1 iter_args(%arg14 = %48) -> (tensor<?x?xf32>) {
        %79 = arith.muli %arg13, %2 : index
        %80 = loom.subview %arg0[%11, %13, %28, %14, %79] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %80, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %81 = loom.bufferize_to_tensor %53[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %82 = loom.subview %arg1[%11, %12, %13, %79] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %82, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %83 = loom.bufferize_to_tensor %57[%2] : memref<?xf16> -> tensor<?xf16>
        %84 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%83 : tensor<?xf16>) outs(%60 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %92 = arith.extf %in : f16 to f32
          linalg.yield %92 : f32
        } -> tensor<?xf32>
        loom.semaphore_give %57 : memref<?xf16>
        %85 = loom.subview %arg2[%11, %12, %13, %79] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %85, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %86 = loom.bufferize_to_tensor %56[%2] : memref<?xf16> -> tensor<?xf16>
        %87 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%86 : tensor<?xf16>) outs(%63 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %92 = arith.extf %in : f16 to f32
          linalg.yield %92 : f32
        } -> tensor<?xf32>
        loom.semaphore_give %56 : memref<?xf16>
        %88 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%81, %22, %84, %87 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%54 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f32, %in_3: f32, %in_4: f32, %out: f16):
          %92 = arith.truncf %cst_0 : f64 to f32
          %93 = arith.mulf %in_3, %92 : f32
          %94 = arith.truncf %cst_0 : f64 to f32
          %95 = arith.mulf %in_2, %94 : f32
          %96 = arith.subf %95, %93 : f32
          %97 = math.powf %cst, %96 : f32
          %98 = arith.extf %in : f16 to f32
          %99 = arith.mulf %98, %97 : f32
          %100 = arith.mulf %99, %in_4 : f32
          %101 = arith.truncf %100 : f32 to f16
          linalg.yield %101 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %62 : memref<?xf32>
        loom.semaphore_give %59 : memref<?xf32>
        %89 = loom.subview %arg3[%11, %27, %12, %33] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        loom.copy %89, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
        %90 = loom.bufferize_to_tensor %65[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %91 = linalg.matmul ins(%88, %90 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        loom.semaphore_give %65 : memref<?x?xf16>
        loom.semaphore_give %53 : memref<?x?xf16>
        scf.yield %91 : tensor<?x?xf32>
      }
      loom.semaphore_give %20 : memref<?xf32>
      %67 = loom.alloc [1] on @L1 : memref<f16>
      %68 = loom.semaphore_take %67 : memref<f16> -> memref<f16>
      %69 = loom.subview %arg6[%12] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      loom.copy %69, %68 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
      %70 = loom.bufferize_to_tensor %68[] : memref<f16> -> tensor<f16>
      %71 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %72 = loom.semaphore_take %71 : memref<?x?xf16> -> memref<?x?xf16>
      %73 = loom.init_tensor %72[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %74 = loom.subview %arg3[%11, %27, %12, %33] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      loom.copy %74, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
      %75 = loom.bufferize_to_tensor %72[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %75, %70 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%73 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %in_2: f16, %in_3: f16, %out: f16):
        %79 = arith.extf %in_3 : f16 to f32
        %80 = arith.extf %in_2 : f16 to f32
        %81 = arith.mulf %80, %79 : f32
        %82 = arith.addf %in, %81 : f32
        %83 = arith.truncf %82 : f32 to f16
        linalg.yield %83 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %68 : memref<f16>
      loom.semaphore_give %44 : memref<?x?xf32>
      %77 = loom.subview %arg7[%11, %27, %12, %33] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %78 = loom.bufferize_to_memref %76 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %78, %77 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      loom.semaphore_give %72 : memref<?x?xf16>
    }
    return
  }
}
