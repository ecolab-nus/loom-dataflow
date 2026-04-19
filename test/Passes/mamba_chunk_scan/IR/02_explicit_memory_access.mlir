module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.442380e+00 : f16
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
      %19 = loom.alloc [%0] on @L1 : memref<?xf16>
      %20 = loom.semaphore_take %19 : memref<?xf16> -> memref<?xf16>
      %21 = loom.init_tensor %20[%0] : memref<?xf16> -> tensor<?xf16>
      %22 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%18 : tensor<?xf16>) outs(%21 : tensor<?xf16>) {
      ^bb0(%in: f16, %out: f16):
        %73 = arith.mulf %in, %cst_1 : f16
        %74 = math.powf %cst, %73 : f16
        linalg.yield %74 : f16
      } -> tensor<?xf16>
      %23 = arith.muli %13, %c256 : index
      %24 = arith.divui %12, %c64 : index
      %25 = loom.alloc [%0, 64] on @L1 : memref<?x64xf16>
      %26 = loom.semaphore_take %25 : memref<?x64xf16> -> memref<?x64xf16>
      %27 = loom.subview %arg4[%11, %23, %24, 0] [1, %0, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      loom.copy %27, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
      %28 = loom.bufferize_to_tensor %26[%0, 64] : memref<?x64xf16> -> tensor<?x64xf16>
      %29 = arith.muli %arg10, %1 : index
      %30 = loom.alloc [%1, 64] on @L1 : memref<?x64xf16>
      %31 = loom.semaphore_take %30 : memref<?x64xf16> -> memref<?x64xf16>
      %32 = loom.subview %arg5[%11, %13, %12, %29, 0] [1, 1, 1, %1, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
      %33 = loom.bufferize_to_tensor %31[%1, 64] : memref<?x64xf16> -> tensor<?x64xf16>
      %34 = loom.alloc [64, %1] on @L1 : memref<64x?xf16>
      %35 = loom.semaphore_take %34 : memref<64x?xf16> -> memref<64x?xf16>
      %36 = loom.init_tensor %35[64, %1] : memref<64x?xf16> -> tensor<64x?xf16>
      %transposed = linalg.transpose ins(%33 : tensor<?x64xf16>) outs(%36 : tensor<64x?xf16>) permutation = [1, 0] 
      loom.semaphore_give %31 : memref<?x64xf16>
      %37 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
      %39 = loom.init_tensor %38[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %40 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
      %41 = loom.init_tensor %40[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %42 = linalg.fill ins(%cst_0 : f16) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %43 = linalg.matmul ins(%28, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %35 : memref<64x?xf16>
      loom.semaphore_give %26 : memref<?x64xf16>
      %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%43, %22 : tensor<?x?xf16>, tensor<?xf16>) outs(%41 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_2: f16, %out: f16):
        %73 = arith.mulf %in, %in_2 : f16
        linalg.yield %73 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %38 : memref<?x?xf16>
      loom.semaphore_give %20 : memref<?xf16>
      %45 = arith.addi %arg9, %c1 : index
      %46 = arith.muli %45, %0 : index
      %47 = arith.ceildivui %46, %2 : index
      %48 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %49 = loom.semaphore_take %48 : memref<?x?xf16> -> memref<?x?xf16>
      %50 = loom.init_tensor %49[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %51 = loom.alloc [%2] on @L1 : memref<?xf16>
      %52 = loom.semaphore_take %51 : memref<?xf16> -> memref<?xf16>
      %53 = loom.alloc [%2] on @L1 : memref<?xf16>
      %54 = loom.semaphore_take %53 : memref<?xf16> -> memref<?xf16>
      %55 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %56 = loom.semaphore_take %55 : memref<?x?xf16> -> memref<?x?xf16>
      %57 = scf.for %arg13 = %c0 to %47 step %c1 iter_args(%arg14 = %44) -> (tensor<?x?xf16>) {
        %73 = arith.muli %arg13, %2 : index
        %74 = loom.subview %arg0[%11, %13, %24, %14, %73] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %74, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %75 = loom.bufferize_to_tensor %49[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %76 = loom.subview %arg1[%11, %12, %13, %73] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %76, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %77 = loom.bufferize_to_tensor %52[%2] : memref<?xf16> -> tensor<?xf16>
        %78 = loom.subview %arg2[%11, %12, %13, %73] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %78, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %79 = loom.bufferize_to_tensor %54[%2] : memref<?xf16> -> tensor<?xf16>
        %80 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %18, %77, %79 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%50 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
          %84 = arith.mulf %in_3, %cst_1 : f16
          %85 = arith.mulf %in_2, %cst_1 : f16
          %86 = arith.subf %85, %84 : f16
          %87 = math.powf %cst, %86 : f16
          %88 = arith.mulf %in, %87 : f16
          %89 = arith.mulf %88, %in_4 : f16
          linalg.yield %89 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %54 : memref<?xf16>
        loom.semaphore_give %52 : memref<?xf16>
        %81 = loom.subview %arg3[%11, %23, %12, %29] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        loom.copy %81, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
        %82 = loom.bufferize_to_tensor %56[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %83 = linalg.matmul ins(%80, %82 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %56 : memref<?x?xf16>
        loom.semaphore_give %49 : memref<?x?xf16>
        scf.yield %83 : tensor<?x?xf16>
      }
      loom.semaphore_give %16 : memref<?xf16>
      %58 = loom.alloc [1] on @L1 : memref<f16>
      %59 = loom.semaphore_take %58 : memref<f16> -> memref<f16>
      %60 = loom.subview %arg6[%12] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      loom.copy %60, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
      %61 = loom.bufferize_to_tensor %59[] : memref<f16> -> tensor<f16>
      %62 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %63 = loom.semaphore_take %62 : memref<?x?xf16> -> memref<?x?xf16>
      %64 = loom.init_tensor %63[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %65 = loom.subview %arg3[%11, %23, %12, %29] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.copy %65, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
      %66 = loom.bufferize_to_tensor %63[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%57, %66, %61 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%64 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
        %73 = arith.mulf %in_2, %in_3 : f16
        %74 = arith.addf %in, %73 : f16
        linalg.yield %74 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %59 : memref<f16>
      loom.semaphore_give %40 : memref<?x?xf16>
      %68 = loom.subview %arg7[%11, %23, %12, %29] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %69 = loom.semaphore_take %62 : memref<?x?xf16> -> memref<?x?xf16>
      %70 = loom.init_tensor %69[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %71 = loom.sync ins(%67 : tensor<?x?xf16>) outs(%70 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %72 = loom.bufferize_to_memref %71 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %72, %68 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %69 : memref<?x?xf16>
      loom.semaphore_give %63 : memref<?x?xf16>
    }
    return
  }
}
