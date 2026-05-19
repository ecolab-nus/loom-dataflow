module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x2048x64xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x64x8x256xf16>, %arg4: memref<2x1x2048x64xf16>, %arg5: memref<64xf16>, %arg6: memref<2x8x64x64x64xf16>, %arg7: memref<2x64x2048x64xf16>) {
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
      %17 = loom.subview %arg3[%11, %12, %13, %14] [1, 1, 1, %0] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
      loom.copy %17, %16 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
      %18 = loom.bufferize_to_tensor %16[%0] : memref<?xf16> -> tensor<?xf16>
      %19 = loom.alloc [%1, %0] on @L1 : memref<?x?xf16>
      %20 = loom.semaphore_take %19 : memref<?x?xf16> -> memref<?x?xf16>
      %21 = loom.init_tensor %20[%1, %0] : memref<?x?xf16> -> tensor<?x?xf16>
      %22 = loom.broadcast ins(%18 : tensor<?xf16>) outs(%21 : tensor<?x?xf16>) dim(0) -> tensor<?x?xf16>
      %23 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %24 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
      %25 = loom.init_tensor %24[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %26 = loom.semaphore_take %23 : memref<?x?xf16> -> memref<?x?xf16>
      %27 = loom.init_tensor %26[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %transposed = linalg.transpose ins(%22 : tensor<?x?xf16>) outs(%25 : tensor<?x?xf16>) permutation = [1, 0] 
      loom.semaphore_give %20 : memref<?x?xf16>
      %28 = arith.divui %12, %c64 : index
      %29 = arith.muli %13, %c256 : index
      %30 = arith.addi %14, %29 : index
      %31 = loom.alloc [%0, 64] on @L1 : memref<?x64xf16>
      %32 = loom.semaphore_take %31 : memref<?x64xf16> -> memref<?x64xf16>
      %33 = loom.subview %arg4[%11, %28, %30, 0] [1, 1, %0, 64] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x1x2048x64xf16> to memref<?x64xf16, strided<[64, 1], offset: ?>>
      loom.copy %33, %32 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x64xf16, strided<[64, 1], offset: ?>> to memref<?x64xf16>
      %34 = loom.bufferize_to_tensor %32[%0, 64] : memref<?x64xf16> -> tensor<?x64xf16>
      %35 = arith.muli %arg10, %1 : index
      %36 = loom.alloc [64, %1] on @L1 : memref<64x?xf16>
      %37 = loom.semaphore_take %36 : memref<64x?xf16> -> memref<64x?xf16>
      %38 = loom.subview %arg6[%11, %13, %12, 0, %35] [1, 1, 1, 64, %1] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x64x64x64xf16> to memref<64x?xf16, strided<[64, 1], offset: ?>>
      loom.copy %38, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<64x?xf16, strided<[64, 1], offset: ?>> to memref<64x?xf16>
      %39 = loom.bufferize_to_tensor %37[64, %1] : memref<64x?xf16> -> tensor<64x?xf16>
      %40 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %41 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
      %42 = loom.init_tensor %41[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %43 = loom.semaphore_take %40 : memref<?x?xf16> -> memref<?x?xf16>
      %44 = loom.init_tensor %43[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %45 = linalg.fill ins(%cst : f16) outs(%44 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %46 = linalg.matmul ins(%34, %39 : tensor<?x64xf16>, tensor<64x?xf16>) outs(%45 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %37 : memref<64x?xf16>
      loom.semaphore_give %32 : memref<?x64xf16>
      %47 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%46, %transposed : tensor<?x?xf16>, tensor<?x?xf16>) outs(%27 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_0: f16, %out: f16):
        %86 = math.exp %in_0 : f16
        %87 = arith.mulf %in, %86 : f16
        linalg.yield %87 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %43 : memref<?x?xf16>
      loom.semaphore_give %24 : memref<?x?xf16>
      %48 = arith.addi %arg9, %c1 : index
      %49 = arith.muli %48, %0 : index
      %50 = arith.ceildivui %49, %2 : index
      %51 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %52 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
      %53 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %54 = loom.semaphore_take %53 : memref<?x?xf16> -> memref<?x?xf16>
      %55 = loom.init_tensor %54[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %56 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %57 = loom.semaphore_take %56 : memref<?x?xf16> -> memref<?x?xf16>
      %58 = loom.init_tensor %57[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %59 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %60 = loom.semaphore_take %59 : memref<?x?xf16> -> memref<?x?xf16>
      %61 = loom.init_tensor %60[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %62 = loom.alloc [%2] on @L1 : memref<?xf16>
      %63 = loom.semaphore_take %62 : memref<?xf16> -> memref<?xf16>
      %64 = loom.alloc [%2] on @L1 : memref<?xf16>
      %65 = loom.semaphore_take %64 : memref<?xf16> -> memref<?xf16>
      %66 = loom.alloc [%2, %0] on @L1 : memref<?x?xf16>
      %67 = loom.semaphore_take %66 : memref<?x?xf16> -> memref<?x?xf16>
      %68 = loom.init_tensor %67[%2, %0] : memref<?x?xf16> -> tensor<?x?xf16>
      %69 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %70 = loom.semaphore_take %69 : memref<?x?xf16> -> memref<?x?xf16>
      %71 = scf.for %arg13 = %c0 to %50 step %c1 iter_args(%arg14 = %47) -> (tensor<?x?xf16>) {
        %86 = arith.muli %arg13, %2 : index
        %87 = loom.subview %arg0[%11, %13, %28, %14, %86] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %87, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %88 = loom.bufferize_to_tensor %52[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
        %89 = loom.subview %arg3[%11, %12, %13, %86] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %89, %63 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %90 = loom.bufferize_to_tensor %63[%2] : memref<?xf16> -> tensor<?xf16>
        %91 = loom.broadcast ins(%18 : tensor<?xf16>) outs(%68 : tensor<?x?xf16>) dim(0) -> tensor<?x?xf16>
        %transposed_0 = linalg.transpose ins(%91 : tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) permutation = [1, 0] 
        loom.semaphore_give %67 : memref<?x?xf16>
        %92 = loom.subview %arg2[%11, %12, %13, %86] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %92, %65 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %93 = loom.bufferize_to_tensor %65[%2] : memref<?xf16> -> tensor<?xf16>
        %94 = loom.broadcast ins(%93 : tensor<?xf16>) outs(%58 : tensor<?x?xf16>) dim(0) -> tensor<?x?xf16>
        loom.semaphore_give %65 : memref<?xf16>
        %95 = loom.broadcast ins(%90 : tensor<?xf16>) outs(%61 : tensor<?x?xf16>) dim(0) -> tensor<?x?xf16>
        loom.semaphore_give %63 : memref<?xf16>
        %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%88, %transposed_0, %95, %94 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%55 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_1: f16, %in_2: f16, %in_3: f16, %out: f16):
          %103 = arith.subf %in_1, %in_2 : f16
          %104 = math.exp %103 : f16
          %105 = arith.mulf %in, %104 : f16
          %106 = arith.mulf %105, %in_3 : f16
          linalg.yield %106 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %60 : memref<?x?xf16>
        loom.semaphore_give %57 : memref<?x?xf16>
        loom.semaphore_give %52 : memref<?x?xf16>
        %97 = arith.addi %86, %29 : index
        %98 = loom.subview %arg1[%11, %12, %97, %35] [1, 1, %2, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x2048x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
        loom.copy %98, %70 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[64, 1], offset: ?>> to memref<?x?xf16>
        %99 = loom.bufferize_to_tensor %70[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %100 = linalg.fill ins(%cst : f16) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
        %101 = linalg.matmul ins(%96, %99 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%100 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %70 : memref<?x?xf16>
        loom.semaphore_give %54 : memref<?x?xf16>
        %102 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %101 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_1: f16, %out: f16):
          %103 = arith.addf %in, %in_1 : f16
          linalg.yield %103 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %41 : memref<?x?xf16>
        scf.yield %102 : tensor<?x?xf16>
      }
      loom.semaphore_give %16 : memref<?xf16>
      %72 = loom.alloc [1] on @L1 : memref<f16>
      %73 = loom.semaphore_take %72 : memref<f16> -> memref<f16>
      %74 = loom.subview %arg5[%12] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
      loom.copy %74, %73 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
      %75 = loom.bufferize_to_tensor %73[] : memref<f16> -> tensor<f16>
      %76 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %77 = loom.semaphore_take %76 : memref<?x?xf16> -> memref<?x?xf16>
      %78 = loom.subview %arg1[%11, %12, %30, %35] [1, 1, %0, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x2048x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
      loom.copy %78, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[64, 1], offset: ?>> to memref<?x?xf16>
      %79 = loom.bufferize_to_tensor %77[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %80 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %81 = loom.semaphore_take %80 : memref<?x?xf16> -> memref<?x?xf16>
      %82 = loom.init_tensor %81[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%71, %79, %75 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%82 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
        %86 = arith.mulf %in_0, %in_1 : f16
        %87 = arith.addf %in, %86 : f16
        linalg.yield %87 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %73 : memref<f16>
      loom.semaphore_give %77 : memref<?x?xf16>
      loom.semaphore_give %26 : memref<?x?xf16>
      %84 = loom.subview %arg7[%11, %12, %30, %35] [1, 1, %0, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x64x2048x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
      %85 = loom.bufferize_to_memref %83 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %85, %84 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
      loom.semaphore_give %81 : memref<?x?xf16>
    }
    return
  }
}
