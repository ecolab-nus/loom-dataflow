module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 2.000000e+00 : f16
    %c0 = arith.constant 0 : index
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
      %19 = arith.muli %13, %c256 : index
      %20 = arith.divui %12, %c64 : index
      %21 = loom.alloc [%0, 64] on @L1 : memref<?x64xf16>
      %22 = loom.semaphore_take %21 : memref<?x64xf16> -> memref<?x64xf16>
      %23 = loom.subview %arg4[%11, %19, %20, 0] [1, %0, 1, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      loom.copy %23, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
      %24 = loom.bufferize_to_tensor %22[%0, 64] : memref<?x64xf16> -> tensor<?x64xf16>
      %25 = arith.muli %arg10, %1 : index
      %26 = loom.alloc [%1, 64] on @L1 : memref<?x64xf16>
      %27 = loom.semaphore_take %26 : memref<?x64xf16> -> memref<?x64xf16>
      %28 = loom.subview %arg5[%11, %13, %12, %25, 0] [1, 1, 1, %1, 64] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      loom.copy %28, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
      %29 = loom.bufferize_to_tensor %27[%1, 64] : memref<?x64xf16> -> tensor<?x64xf16>
      %30 = loom.alloc [64, %1] on @L1 : memref<64x?xf16>
      %31 = loom.semaphore_take %30 : memref<64x?xf16> -> memref<64x?xf16>
      %32 = loom.init_tensor %31[64, %1] : memref<64x?xf16> -> tensor<64x?xf16>
      %transposed = linalg.transpose ins(%29 : tensor<?x64xf16>) outs(%32 : tensor<64x?xf16>) permutation = [1, 0] 
      loom.semaphore_give %27 : memref<?x64xf16>
      %33 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %34 = loom.semaphore_take %33 : memref<?x?xf16> -> memref<?x?xf16>
      %35 = loom.init_tensor %34[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %36 = loom.semaphore_take %33 : memref<?x?xf16> -> memref<?x?xf16>
      %37 = loom.init_tensor %36[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %38 = linalg.fill ins(%cst_0 : f16) outs(%35 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %39 = linalg.matmul ins(%24, %transposed : tensor<?x64xf16>, tensor<64x?xf16>) outs(%38 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %31 : memref<64x?xf16>
      loom.semaphore_give %22 : memref<?x64xf16>
      %40 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%39, %18 : tensor<?x?xf16>, tensor<?xf16>) outs(%37 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_2: f16, %out: f16):
        %71 = arith.mulf %in_2, %cst_1 : f16
        %72 = math.powf %cst, %71 : f16
        %73 = arith.mulf %in, %72 : f16
        linalg.yield %73 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %34 : memref<?x?xf16>
      %41 = arith.addi %arg9, %c1 : index
      %42 = arith.muli %41, %0 : index
      %43 = arith.ceildivui %42, %2 : index
      %44 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
      %46 = loom.init_tensor %45[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %47 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
      %48 = loom.init_tensor %47[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %49 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %50 = loom.semaphore_take %49 : memref<?x?xf16> -> memref<?x?xf16>
      %51 = loom.init_tensor %50[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %52 = loom.alloc [%2] on @L1 : memref<?xf16>
      %53 = loom.semaphore_take %52 : memref<?xf16> -> memref<?xf16>
      %54 = loom.alloc [%2] on @L1 : memref<?xf16>
      %55 = loom.semaphore_take %54 : memref<?xf16> -> memref<?xf16>
      %56 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
      %58 = scf.for %arg13 = %c0 to %43 step %c1 iter_args(%arg14 = %40) -> (tensor<?x?xf16>) {
        %71 = arith.muli %arg13, %2 : index
        %72 = loom.subview %arg0[%11, %13, %20, %14, %71] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %72, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %73 = loom.bufferize_to_tensor %50[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %74 = loom.subview %arg1[%11, %12, %13, %71] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %74, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %75 = loom.bufferize_to_tensor %53[%2] : memref<?xf16> -> tensor<?xf16>
        %76 = loom.subview %arg2[%11, %12, %13, %71] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %76, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %77 = loom.bufferize_to_tensor %55[%2] : memref<?xf16> -> tensor<?xf16>
        %78 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%73, %18, %75, %77 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%51 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
          %84 = arith.mulf %in_3, %cst_1 : f16
          %85 = arith.mulf %in_2, %cst_1 : f16
          %86 = arith.subf %85, %84 : f16
          %87 = math.powf %cst, %86 : f16
          %88 = arith.mulf %in, %87 : f16
          %89 = arith.mulf %88, %in_4 : f16
          linalg.yield %89 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %55 : memref<?xf16>
        loom.semaphore_give %53 : memref<?xf16>
        %79 = loom.subview %arg3[%11, %19, %12, %25] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
        loom.copy %79, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
        %80 = loom.bufferize_to_tensor %57[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %81 = linalg.fill ins(%cst_0 : f16) outs(%48 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %82 = linalg.matmul ins(%78, %80 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%81 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %57 : memref<?x?xf16>
        loom.semaphore_give %50 : memref<?x?xf16>
        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %82 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f16, %out: f16):
          %84 = arith.addf %in, %in_2 : f16
          linalg.yield %84 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %47 : memref<?x?xf16>
        scf.yield %83 : tensor<?x?xf16>
      }
      loom.semaphore_give %16 : memref<?xf16>
      %59 = loom.alloc [1] on @L1 : memref<f16>
      %60 = loom.semaphore_take %59 : memref<f16> -> memref<f16>
      %61 = loom.subview %arg6[%12] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      loom.copy %61, %60 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
      %62 = loom.bufferize_to_tensor %60[] : memref<f16> -> tensor<f16>
      %63 = loom.subview %arg3[%11, %19, %12, %25] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.copy %63, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[4096, 1], offset: ?>> to memref<?x?xf16>
      %64 = loom.bufferize_to_tensor %45[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %65 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%58, %64, %62 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%46 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
        %71 = arith.mulf %in_2, %in_3 : f16
        %72 = arith.addf %in, %71 : f16
        linalg.yield %72 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %60 : memref<f16>
      loom.semaphore_give %36 : memref<?x?xf16>
      %66 = loom.subview %arg7[%11, %19, %12, %25] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x64x64xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      %67 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
      %68 = loom.init_tensor %67[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %69 = loom.sync ins(%65 : tensor<?x?xf16>) outs(%68 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %70 = loom.bufferize_to_memref %69 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %70, %66 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[4096, 1], offset: ?>>
      loom.semaphore_give %67 : memref<?x?xf16>
      loom.semaphore_give %45 : memref<?x?xf16>
    }
    return
  }
}
