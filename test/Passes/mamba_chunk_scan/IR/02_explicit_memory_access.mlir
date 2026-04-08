module attributes {loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<1x8x1x256x256xf16>, %arg1: memref<1x16x8x256xf16>, %arg2: memref<1x16x8x256xf16>, %arg3: memref<1x2048x16x64xf16>, %arg4: memref<1x2048x1x?xf16>, %arg5: memref<1x8x16x64x?xf16>, %arg6: memref<16xf16>, %arg7: memref<1x2048x16x64xf16>) {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f32
    %cst_0 = arith.constant 1.44269504 : f64
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %0 = loom.sym @tile_m {upper_bound = 256 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 64 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_h {upper_bound = 16 : index} : index
    %4 = loom.sym @tile_c {upper_bound = 8 : index} : index
    affine.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (0, 0, 0, 0, 0) to (16 ceildiv symbol(%3), 256 ceildiv symbol(%0), 64 ceildiv symbol(%1), 1, 8 ceildiv symbol(%4)) {
      %5 = arith.muli %arg8, %3 : index
      %6 = arith.muli %arg12, %4 : index
      %7 = arith.muli %arg9, %0 : index
      %8 = loom.alloc [1, 1, 1, %0] on @L1 : memref<1x1x1x?xf16>
      %9 = loom.semaphore_take %8 : memref<1x1x1x?xf16> -> memref<1x1x1x?xf16>
      %10 = loom.subview %arg1[%arg11, %5, %6, %7] [1, 1, 1, %0] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x16x8x256xf16> to memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>>
      loom.copy %10, %9 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>> to memref<1x1x1x?xf16>
      %11 = loom.bufferize_to_tensor %9[1, 1, 1, %0] : memref<1x1x1x?xf16> -> tensor<?xf16>
      %12 = loom.alloc [%0] on @L1 : memref<?xf32>
      %13 = loom.semaphore_take %12 : memref<?xf32> -> memref<?xf32>
      %14 = loom.init_tensor %13[%0] : memref<?xf32> -> tensor<?xf32>
      %15 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%11 : tensor<?xf16>) outs(%14 : tensor<?xf32>) {
      ^bb0(%in: f16, %out: f32):
        %76 = arith.extf %in : f16 to f32
        linalg.yield %76 : f32
      } -> tensor<?xf32>
      loom.semaphore_give %9 : memref<1x1x1x?xf16>
      %16 = loom.alloc [%0] on @L1 : memref<?xf32>
      %17 = loom.semaphore_take %16 : memref<?xf32> -> memref<?xf32>
      %18 = loom.init_tensor %17[%0] : memref<?xf32> -> tensor<?xf32>
      %19 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%15 : tensor<?xf32>) outs(%18 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %76 = arith.truncf %cst_0 : f64 to f32
        %77 = arith.mulf %in, %76 : f32
        %78 = math.powf %cst, %77 : f32
        linalg.yield %78 : f32
      } -> tensor<?xf32>
      %20 = arith.muli %6, %c256 : index
      %21 = affine.apply affine_map<(d0) -> (d0 floordiv 16)>(%5)
      %dim = memref.dim %arg4, %c3 : memref<1x2048x1x?xf16>
      %22 = loom.alloc [1, %0, 1, %dim] on @L1 : memref<1x?x1x?xf16>
      %23 = loom.semaphore_take %22 : memref<1x?x1x?xf16> -> memref<1x?x1x?xf16>
      %24 = loom.subview %arg4[%arg11, %20, %21, 0] [1, %0, 1, %dim] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x1x?xf16> to memref<1x?x1x?xf16, strided<[?, ?, ?, 1], offset: ?>>
      loom.copy %24, %23 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x?x1x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x?x1x?xf16>
      %25 = loom.bufferize_to_tensor %23[1, %0, 1, %dim] : memref<1x?x1x?xf16> -> tensor<?x?xf16>
      %26 = arith.muli %arg10, %1 : index
      %dim_2 = memref.dim %arg5, %c4 : memref<1x8x16x64x?xf16>
      %27 = loom.alloc [1, 1, 1, %1, %dim_2] on @L1 : memref<1x1x1x?x?xf16>
      %28 = loom.semaphore_take %27 : memref<1x1x1x?x?xf16> -> memref<1x1x1x?x?xf16>
      %29 = loom.subview %arg5[%arg11, %6, %5, %26, 0] [1, 1, 1, %1, %dim_2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x8x16x64x?xf16> to memref<1x1x1x?x?xf16, strided<[?, ?, ?, ?, 1], offset: ?>>
      loom.copy %29, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?x?xf16, strided<[?, ?, ?, ?, 1], offset: ?>> to memref<1x1x1x?x?xf16>
      %30 = loom.bufferize_to_tensor %28[1, 1, 1, %1, %dim_2] : memref<1x1x1x?x?xf16> -> tensor<?x?xf16>
      %31 = loom.alloc [%dim_2, %1] on @L1 : memref<?x?xf16>
      %32 = loom.semaphore_take %31 : memref<?x?xf16> -> memref<?x?xf16>
      %33 = loom.init_tensor %32[%dim_2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %transposed = linalg.transpose ins(%30 : tensor<?x?xf16>) outs(%33 : tensor<?x?xf16>) permutation = [1, 0] 
      loom.semaphore_give %28 : memref<1x1x1x?x?xf16>
      %34 = loom.alloc [%0, %1] on @L1 : memref<?x?xf32>
      %35 = loom.semaphore_take %34 : memref<?x?xf32> -> memref<?x?xf32>
      %36 = loom.init_tensor %35[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %37 = loom.semaphore_take %34 : memref<?x?xf32> -> memref<?x?xf32>
      %38 = loom.init_tensor %37[%0, %1] : memref<?x?xf32> -> tensor<?x?xf32>
      %39 = linalg.fill ins(%cst_1 : f32) outs(%36 : tensor<?x?xf32>) -> tensor<?x?xf32>
      %40 = linalg.matmul ins(%25, %transposed : tensor<?x?xf16>, tensor<?x?xf16>) outs(%39 : tensor<?x?xf32>) -> tensor<?x?xf32>
      loom.semaphore_give %32 : memref<?x?xf16>
      loom.semaphore_give %23 : memref<1x?x1x?xf16>
      %41 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%40, %19 : tensor<?x?xf32>, tensor<?xf32>) outs(%38 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %76 = arith.mulf %in, %in_3 : f32
        linalg.yield %76 : f32
      } -> tensor<?x?xf32>
      loom.semaphore_give %35 : memref<?x?xf32>
      loom.semaphore_give %17 : memref<?xf32>
      %42 = arith.addi %arg9, %c1 : index
      %43 = arith.muli %42, %0 : index
      %44 = affine.apply affine_map<(d0)[s0] -> (d0 ceildiv s0)>(%43)[%2]
      %45 = loom.alloc [1, 1, 1, %0, %2] on @L1 : memref<1x1x1x?x?xf16>
      %46 = loom.semaphore_take %45 : memref<1x1x1x?x?xf16> -> memref<1x1x1x?x?xf16>
      %47 = loom.alloc [1, 1, 1, %2] on @L1 : memref<1x1x1x?xf16>
      %48 = loom.semaphore_take %47 : memref<1x1x1x?xf16> -> memref<1x1x1x?xf16>
      %49 = loom.semaphore_take %47 : memref<1x1x1x?xf16> -> memref<1x1x1x?xf16>
      %50 = loom.alloc [%2] on @L1 : memref<?xf32>
      %51 = loom.semaphore_take %50 : memref<?xf32> -> memref<?xf32>
      %52 = loom.init_tensor %51[%2] : memref<?xf32> -> tensor<?xf32>
      %53 = loom.alloc [%2] on @L1 : memref<?xf32>
      %54 = loom.semaphore_take %53 : memref<?xf32> -> memref<?xf32>
      %55 = loom.init_tensor %54[%2] : memref<?xf32> -> tensor<?xf32>
      %56 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
      %58 = loom.init_tensor %57[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %59 = loom.alloc [1, %2, 1, %1] on @L1 : memref<1x?x1x?xf16>
      %60 = loom.semaphore_take %59 : memref<1x?x1x?xf16> -> memref<1x?x1x?xf16>
      %61 = scf.for %arg13 = %c0 to %44 step %c1 iter_args(%arg14 = %41) -> (tensor<?x?xf32>) {
        %76 = arith.muli %arg13, %2 : index
        %77 = loom.subview %arg0[%arg11, %6, %21, %7, %76] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x8x1x256x256xf16> to memref<1x1x1x?x?xf16, strided<[524288, 65536, 65536, 256, 1], offset: ?>>
        loom.copy %77, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?x?xf16, strided<[524288, 65536, 65536, 256, 1], offset: ?>> to memref<1x1x1x?x?xf16>
        %78 = loom.bufferize_to_tensor %46[1, 1, 1, %0, %2] : memref<1x1x1x?x?xf16> -> tensor<?x?xf16>
        %79 = loom.subview %arg1[%arg11, %5, %6, %76] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x16x8x256xf16> to memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>>
        loom.copy %79, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>> to memref<1x1x1x?xf16>
        %80 = loom.bufferize_to_tensor %49[1, 1, 1, %2] : memref<1x1x1x?xf16> -> tensor<?xf16>
        %81 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%80 : tensor<?xf16>) outs(%52 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %89 = arith.extf %in : f16 to f32
          linalg.yield %89 : f32
        } -> tensor<?xf32>
        loom.semaphore_give %49 : memref<1x1x1x?xf16>
        %82 = loom.subview %arg2[%arg11, %5, %6, %76] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x16x8x256xf16> to memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>>
        loom.copy %82, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x1x1x?xf16, strided<[32768, 2048, 256, 1], offset: ?>> to memref<1x1x1x?xf16>
        %83 = loom.bufferize_to_tensor %48[1, 1, 1, %2] : memref<1x1x1x?xf16> -> tensor<?xf16>
        %84 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%83 : tensor<?xf16>) outs(%55 : tensor<?xf32>) {
        ^bb0(%in: f16, %out: f32):
          %89 = arith.extf %in : f16 to f32
          linalg.yield %89 : f32
        } -> tensor<?xf32>
        loom.semaphore_give %48 : memref<1x1x1x?xf16>
        %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%78, %15, %81, %84 : tensor<?x?xf16>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%58 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_3: f32, %in_4: f32, %in_5: f32, %out: f16):
          %89 = arith.truncf %cst_0 : f64 to f32
          %90 = arith.mulf %in_4, %89 : f32
          %91 = arith.truncf %cst_0 : f64 to f32
          %92 = arith.mulf %in_3, %91 : f32
          %93 = arith.subf %92, %90 : f32
          %94 = math.powf %cst, %93 : f32
          %95 = arith.extf %in : f16 to f32
          %96 = arith.mulf %95, %94 : f32
          %97 = arith.mulf %96, %in_5 : f32
          %98 = arith.truncf %97 : f32 to f16
          linalg.yield %98 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %54 : memref<?xf32>
        loom.semaphore_give %51 : memref<?xf32>
        loom.semaphore_give %46 : memref<1x1x1x?x?xf16>
        %86 = loom.subview %arg3[%arg11, %20, %5, %26] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x16x64xf16> to memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>>
        loom.copy %86, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>> to memref<1x?x1x?xf16>
        %87 = loom.bufferize_to_tensor %60[1, %2, 1, %1] : memref<1x?x1x?xf16> -> tensor<?x?xf16>
        %88 = linalg.matmul ins(%85, %87 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf32>) -> tensor<?x?xf32>
        loom.semaphore_give %60 : memref<1x?x1x?xf16>
        loom.semaphore_give %57 : memref<?x?xf16>
        scf.yield %88 : tensor<?x?xf32>
      }
      loom.semaphore_give %13 : memref<?xf32>
      %62 = loom.alloc [1] on @L1 : memref<1xf16>
      %63 = loom.semaphore_take %62 : memref<1xf16> -> memref<1xf16>
      %64 = loom.subview %arg6[%5] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<1xf16, strided<[1], offset: ?>>
      loom.copy %64, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1xf16, strided<[1], offset: ?>> to memref<1xf16>
      %65 = loom.bufferize_to_tensor %63[1] : memref<1xf16> -> tensor<f16>
      %66 = loom.alloc [1, %0, 1, %1] on @L1 : memref<1x?x1x?xf16>
      %67 = loom.semaphore_take %66 : memref<1x?x1x?xf16> -> memref<1x?x1x?xf16>
      %68 = loom.subview %arg3[%arg11, %20, %5, %26] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x16x64xf16> to memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>>
      loom.copy %68, %67 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>> to memref<1x?x1x?xf16>
      %69 = loom.bufferize_to_tensor %67[1, %0, 1, %1] : memref<1x?x1x?xf16> -> tensor<?x?xf16>
      %70 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %71 = loom.semaphore_take %70 : memref<?x?xf16> -> memref<?x?xf16>
      %72 = loom.init_tensor %71[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %73 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%61, %69, %65 : tensor<?x?xf32>, tensor<?x?xf16>, tensor<f16>) outs(%72 : tensor<?x?xf16>) {
      ^bb0(%in: f32, %in_3: f16, %in_4: f16, %out: f16):
        %76 = arith.extf %in_4 : f16 to f32
        %77 = arith.extf %in_3 : f16 to f32
        %78 = arith.mulf %77, %76 : f32
        %79 = arith.addf %in, %78 : f32
        %80 = arith.truncf %79 : f32 to f16
        linalg.yield %80 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %67 : memref<1x?x1x?xf16>
      loom.semaphore_give %63 : memref<1xf16>
      loom.semaphore_give %37 : memref<?x?xf32>
      %74 = loom.subview %arg7[%arg11, %20, %5, %26] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<1x2048x16x64xf16> to memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>>
      %75 = loom.bufferize_to_memref %73 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %75, %74 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<1x?x1x?xf16, strided<[2097152, 1024, 64, 1], offset: ?>>
      loom.semaphore_give %71 : memref<?x?xf16>
    }
    return
  }
}
