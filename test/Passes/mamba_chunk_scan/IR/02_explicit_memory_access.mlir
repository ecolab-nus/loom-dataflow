module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 2.000000e+00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %cst_1 = arith.constant 1.442380e+00 : f16
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
      %19 = loom.alloc [%0] on @L1 : memref<?xf16>
      %20 = loom.semaphore_take %19 : memref<?xf16> -> memref<?xf16>
      %21 = loom.init_tensor %20[%0] : memref<?xf16> -> tensor<?xf16>
      %22 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%18 : tensor<?xf16>) outs(%21 : tensor<?xf16>) {
      ^bb0(%in: f16, %out: f16):
        %66 = arith.mulf %in, %cst_1 : f16
        %67 = math.powf %cst, %66 : f16
        linalg.yield %67 : f16
      } -> tensor<?xf16>
      %23 = arith.muli %13, %c256 : index
      %24 = arith.divui %12, %c16 : index
      %25 = loom.alloc [%0, 16] on @L1 : memref<?x16xf16>
      %26 = loom.semaphore_take %25 : memref<?x16xf16> -> memref<?x16xf16>
      %27 = loom.subview %arg4[%11, %23, %24, 0] [1, %0, 1, 16] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      loom.copy %27, %26 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
      %28 = loom.bufferize_to_tensor %26[%0, 16] : memref<?x16xf16> -> tensor<?x16xf16>
      %29 = arith.muli %arg10, %1 : index
      %30 = loom.alloc [%1, 16] on @L1 : memref<?x16xf16>
      %31 = loom.semaphore_take %30 : memref<?x16xf16> -> memref<?x16xf16>
      %32 = loom.subview %arg5[%11, %13, %12, %29, 0] [1, 1, 1, %1, 16] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
      loom.copy %32, %31 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<?x16xf16>
      %33 = loom.bufferize_to_tensor %31[%1, 16] : memref<?x16xf16> -> tensor<?x16xf16>
      %34 = loom.alloc [16, %1] on @L1 : memref<16x?xf16>
      %35 = loom.semaphore_take %34 : memref<16x?xf16> -> memref<16x?xf16>
      %36 = loom.init_tensor %35[16, %1] : memref<16x?xf16> -> tensor<16x?xf16>
      %transposed = linalg.transpose ins(%33 : tensor<?x16xf16>) outs(%36 : tensor<16x?xf16>) permutation = [1, 0] 
      loom.semaphore_give %31 : memref<?x16xf16>
      %37 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %38 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
      %39 = loom.init_tensor %38[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %40 = loom.semaphore_take %37 : memref<?x?xf16> -> memref<?x?xf16>
      %41 = loom.init_tensor %40[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %42 = linalg.fill ins(%cst_0 : f16) outs(%39 : tensor<?x?xf16>) -> tensor<?x?xf16>
      %43 = linalg.matmul ins(%28, %transposed : tensor<?x16xf16>, tensor<16x?xf16>) outs(%42 : tensor<?x?xf16>) -> tensor<?x?xf16>
      loom.semaphore_give %35 : memref<16x?xf16>
      loom.semaphore_give %26 : memref<?x16xf16>
      %44 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%43, %22 : tensor<?x?xf16>, tensor<?xf16>) outs(%41 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_2: f16, %out: f16):
        %66 = arith.mulf %in, %in_2 : f16
        linalg.yield %66 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %38 : memref<?x?xf16>
      loom.semaphore_give %20 : memref<?xf16>
      %45 = arith.addi %arg9, %c1 : index
      %46 = arith.muli %45, %0 : index
      %47 = arith.ceildivui %46, %2 : index
      %48 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %49 = loom.semaphore_take %48 : memref<?x?xf16> -> memref<?x?xf16>
      %50 = loom.init_tensor %49[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %51 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %52 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
      %53 = scf.for %arg13 = %c0 to %47 step %c1 iter_args(%arg14 = %44) -> (tensor<?x?xf16>) {
        %66 = arith.muli %arg13, %2 : index
        %67 = arith.addi %66, %2 : index
        %68 = arith.cmpi ult, %67, %46 : index
        %69 = arith.select %68, %67, %46 : index
        %70 = arith.subi %69, %66 : index
        %71 = loom.alloc [%0, %70] on @L1 : memref<?x?xf16>
        %72 = loom.semaphore_take %71 : memref<?x?xf16> -> memref<?x?xf16>
        %73 = loom.subview %arg0[%11, %13, %24, %14, %66] [1, 1, 1, %0, %70] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %73, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16>
        %74 = loom.bufferize_to_tensor %72[%0, %70] : memref<?x?xf16> -> tensor<?x?xf16>
        %75 = loom.alloc [%70] on @L1 : memref<?xf16>
        %76 = loom.semaphore_take %75 : memref<?xf16> -> memref<?xf16>
        %77 = loom.subview %arg1[%11, %12, %13, %66] [1, 1, 1, %70] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %77, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %78 = loom.bufferize_to_tensor %76[%70] : memref<?xf16> -> tensor<?xf16>
        %79 = loom.alloc [%70] on @L1 : memref<?xf16>
        %80 = loom.semaphore_take %79 : memref<?xf16> -> memref<?xf16>
        %81 = loom.subview %arg2[%11, %12, %13, %66] [1, 1, 1, %70] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
        loom.copy %81, %80 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
        %82 = loom.bufferize_to_tensor %80[%70] : memref<?xf16> -> tensor<?xf16>
        %83 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%74, %18, %78, %82 : tensor<?x?xf16>, tensor<?xf16>, tensor<?xf16>, tensor<?xf16>) outs(%50 : tensor<?x?xf16>) {
        ^bb0(%in: f16, %in_2: f16, %in_3: f16, %in_4: f16, %out: f16):
          %87 = arith.mulf %in_3, %cst_1 : f16
          %88 = arith.mulf %in_2, %cst_1 : f16
          %89 = arith.subf %88, %87 : f16
          %90 = math.powf %cst, %89 : f16
          %91 = arith.mulf %in, %90 : f16
          %92 = arith.mulf %91, %in_4 : f16
          linalg.yield %92 : f16
        } -> tensor<?x?xf16>
        loom.semaphore_give %80 : memref<?xf16>
        loom.semaphore_give %76 : memref<?xf16>
        loom.semaphore_give %72 : memref<?x?xf16>
        %84 = loom.subview %arg3[%11, %23, %12, %29] [1, %2, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
        loom.copy %84, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
        %85 = loom.bufferize_to_tensor %52[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
        %86 = linalg.matmul ins(%83, %85 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%arg14 : tensor<?x?xf16>) -> tensor<?x?xf16>
        loom.semaphore_give %52 : memref<?x?xf16>
        loom.semaphore_give %49 : memref<?x?xf16>
        scf.yield %86 : tensor<?x?xf16>
      }
      loom.semaphore_give %16 : memref<?xf16>
      %54 = loom.alloc [1] on @L1 : memref<f16>
      %55 = loom.semaphore_take %54 : memref<f16> -> memref<f16>
      %56 = loom.subview %arg6[%12] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
      loom.copy %56, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
      %57 = loom.bufferize_to_tensor %55[] : memref<f16> -> tensor<f16>
      %58 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %59 = loom.semaphore_take %58 : memref<?x?xf16> -> memref<?x?xf16>
      %60 = loom.init_tensor %59[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %61 = loom.subview %arg3[%11, %23, %12, %29] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      loom.copy %61, %59 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<?x?xf16>
      %62 = loom.bufferize_to_tensor %59[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %63 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%53, %62, %57 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%60 : tensor<?x?xf16>) {
      ^bb0(%in: f16, %in_2: f16, %in_3: f16, %out: f16):
        %66 = arith.mulf %in_2, %in_3 : f16
        %67 = arith.addf %in, %66 : f16
        linalg.yield %67 : f16
      } -> tensor<?x?xf16>
      loom.semaphore_give %55 : memref<f16>
      loom.semaphore_give %40 : memref<?x?xf16>
      %64 = loom.subview %arg7[%11, %23, %12, %29] [1, %0, 1, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      %65 = loom.bufferize_to_memref %63 : tensor<?x?xf16> -> memref<?x?xf16>
      loom.copy %65, %64 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
      loom.semaphore_give %59 : memref<?x?xf16>
    }
    return
  }
}
