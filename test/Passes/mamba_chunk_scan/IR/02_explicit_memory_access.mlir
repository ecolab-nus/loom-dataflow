module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 32 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 192 : index}, loom.tile_n = {is_reduction = false, upper_bound = 160 : index}} {
  func.func @helion_mamba2_chunk_scan_kernel(%arg0: memref<2x8x4x192x192xf16>, %arg1: memref<2x32x1536x160xf16>, %arg2: memref<2x32x8x192xf16>, %arg3: memref<2x32x8x192xf16>, %arg4: memref<2x4x1536x128xf16>, %arg5: memref<32xf16>, %arg6: memref<2x8x32x128x160xf16>, %arg7: memref<2x32x1536x160xf16>) {
    %cst = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c160 = arith.constant 160 : index
    %c192 = arith.constant 192 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %0 = loom.sym @tile_m {upper_bound = 192 : index} : index
    %1 = loom.sym @tile_n {upper_bound = 160 : index} : index
    %2 = loom.sym @tile_k {upper_bound = 8192 : index} : index
    %3 = loom.sym @tile_c {upper_bound = 8 : index} : index
    %4 = loom.sym @tile_b {upper_bound = 2 : index} : index
    %5 = loom.sym @tile_h {upper_bound = 32 : index} : index
    %6 = arith.ceildivui %c192, %0 : index
    %7 = arith.ceildivui %c160, %1 : index
    %8 = arith.ceildivui %c8, %3 : index
    affine.parallel (%arg8, %arg9, %arg10) = (0, 0, 0) to (symbol(%6), symbol(%7), symbol(%8)) {
      %9 = arith.ceildivui %c2, %4 : index
      %10 = loom.alloc [%0] on @L1 : memref<?xf16>
      %11 = loom.semaphore_take %10 : memref<?xf16> -> memref<?xf16>
      %12 = loom.alloc [%1, %0] on @L1 : memref<?x?xf16>
      %13 = loom.semaphore_take %12 : memref<?x?xf16> -> memref<?x?xf16>
      %14 = loom.init_tensor %13[%1, %0] : memref<?x?xf16> -> tensor<?x?xf16>
      %15 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %16 = loom.semaphore_take %15 : memref<?x?xf16> -> memref<?x?xf16>
      %17 = loom.init_tensor %16[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %18 = loom.semaphore_take %15 : memref<?x?xf16> -> memref<?x?xf16>
      %19 = loom.init_tensor %18[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %20 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %21 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
      %22 = loom.semaphore_take %20 : memref<?x?xf16> -> memref<?x?xf16>
      %23 = loom.init_tensor %22[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %24 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %25 = loom.semaphore_take %24 : memref<?x?xf16> -> memref<?x?xf16>
      %26 = loom.init_tensor %25[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %27 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %28 = loom.semaphore_take %27 : memref<?x?xf16> -> memref<?x?xf16>
      %29 = loom.alloc [%0, %1] on @L1 : memref<?x?xf16>
      %30 = loom.semaphore_take %29 : memref<?x?xf16> -> memref<?x?xf16>
      %31 = loom.init_tensor %30[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
      %32 = loom.alloc [%0, 128] on @L1 : memref<?x128xf16>
      %33 = loom.semaphore_take %32 : memref<?x128xf16> -> memref<?x128xf16>
      %34 = loom.alloc [128, %1] on @L1 : memref<128x?xf16>
      %35 = loom.semaphore_take %34 : memref<128x?xf16> -> memref<128x?xf16>
      %36 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %37 = loom.semaphore_take %36 : memref<?x?xf16> -> memref<?x?xf16>
      %38 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %39 = loom.semaphore_take %38 : memref<?x?xf16> -> memref<?x?xf16>
      %40 = loom.init_tensor %39[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %41 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %42 = loom.semaphore_take %41 : memref<?x?xf16> -> memref<?x?xf16>
      %43 = loom.init_tensor %42[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %44 = loom.alloc [%0, %2] on @L1 : memref<?x?xf16>
      %45 = loom.semaphore_take %44 : memref<?x?xf16> -> memref<?x?xf16>
      %46 = loom.init_tensor %45[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
      %47 = loom.alloc [%2] on @L1 : memref<?xf16>
      %48 = loom.semaphore_take %47 : memref<?xf16> -> memref<?xf16>
      %49 = loom.alloc [%2] on @L1 : memref<?xf16>
      %50 = loom.semaphore_take %49 : memref<?xf16> -> memref<?xf16>
      %51 = loom.alloc [%2, %0] on @L1 : memref<?x?xf16>
      %52 = loom.semaphore_take %51 : memref<?x?xf16> -> memref<?x?xf16>
      %53 = loom.init_tensor %52[%2, %0] : memref<?x?xf16> -> tensor<?x?xf16>
      %54 = loom.alloc [%2, %1] on @L1 : memref<?x?xf16>
      %55 = loom.semaphore_take %54 : memref<?x?xf16> -> memref<?x?xf16>
      %56 = loom.alloc [1] on @L1 : memref<f16>
      %57 = loom.semaphore_take %56 : memref<f16> -> memref<f16>
      scf.for %arg11 = %c0 to %9 step %c1 {
        %58 = arith.ceildivui %c32, %5 : index
        scf.for %arg12 = %c0 to %58 step %c1 {
          %59 = arith.muli %arg11, %4 : index
          %60 = arith.muli %arg12, %5 : index
          %61 = arith.muli %arg10, %3 : index
          %62 = arith.muli %arg8, %0 : index
          %63 = loom.subview %arg3[%59, %60, %61, %62] [1, 1, 1, %0] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x32x8x192xf16> to memref<?xf16, strided<[1], offset: ?>>
          loom.copy %63, %11 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
          %64 = loom.bufferize_to_tensor %11[%0] : memref<?xf16> -> tensor<?xf16>
          %65 = loom.broadcast ins(%64 : tensor<?xf16>) outs(%14 : tensor<?x?xf16>) dim(0) -> tensor<?x?xf16>
          %transposed = linalg.transpose ins(%65 : tensor<?x?xf16>) outs(%19 : tensor<?x?xf16>) permutation = [1, 0] 
          loom.semaphore_give %13 : memref<?x?xf16>
          %66 = arith.divui %60, %c8 : index
          %67 = arith.muli %61, %c192 : index
          %68 = arith.addi %62, %67 : index
          %69 = loom.subview %arg4[%59, %66, %68, 0] [1, 1, %0, 128] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x4x1536x128xf16> to memref<?x128xf16, strided<[128, 1], offset: ?>>
          loom.copy %69, %33 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x128xf16, strided<[128, 1], offset: ?>> to memref<?x128xf16>
          %70 = loom.bufferize_to_tensor %33[%0, 128] : memref<?x128xf16> -> tensor<?x128xf16>
          %71 = arith.muli %arg9, %1 : index
          %72 = loom.subview %arg6[%59, %61, %60, 0, %71] [1, 1, 1, 128, %1] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x32x128x160xf16> to memref<128x?xf16, strided<[160, 1], offset: ?>>
          loom.copy %72, %35 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<128x?xf16, strided<[160, 1], offset: ?>> to memref<128x?xf16>
          %73 = loom.bufferize_to_tensor %35[128, %1] : memref<128x?xf16> -> tensor<128x?xf16>
          %74 = linalg.fill ins(%cst : f16) outs(%23 : tensor<?x?xf16>) -> tensor<?x?xf16>
          %75 = linalg.matmul ins(%70, %73 : tensor<?x128xf16>, tensor<128x?xf16>) outs(%74 : tensor<?x?xf16>) -> tensor<?x?xf16>
          loom.semaphore_give %35 : memref<128x?xf16>
          loom.semaphore_give %33 : memref<?x128xf16>
          loom.semaphore_give %22 : memref<?x?xf16>
          %76 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%75, %transposed : tensor<?x?xf16>, tensor<?x?xf16>) outs(%26 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_0: f16, %out: f16):
            %88 = math.exp %in_0 : f16
            %89 = arith.mulf %in, %88 : f16
            linalg.yield %89 : f16
          } -> tensor<?x?xf16>
          %77 = arith.addi %arg8, %c1 : index
          %78 = arith.muli %77, %0 : index
          %79 = arith.ceildivui %78, %2 : index
          %80 = scf.for %arg13 = %c0 to %79 step %c1 iter_args(%arg14 = %76) -> (tensor<?x?xf16>) {
            %88 = arith.muli %arg13, %2 : index
            %89 = loom.subview %arg0[%59, %61, %66, %62, %88] [1, 1, 1, %0, %2] [1, 1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x8x4x192x192xf16> to memref<?x?xf16, strided<[192, 1], offset: ?>>
            loom.copy %89, %37 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[192, 1], offset: ?>> to memref<?x?xf16>
            %90 = loom.bufferize_to_tensor %37[%0, %2] : memref<?x?xf16> -> tensor<?x?xf16>
            %91 = loom.subview %arg3[%59, %60, %61, %88] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x32x8x192xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %91, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
            %92 = loom.bufferize_to_tensor %48[%2] : memref<?xf16> -> tensor<?xf16>
            %93 = loom.broadcast ins(%64 : tensor<?xf16>) outs(%53 : tensor<?x?xf16>) dim(0) -> tensor<?x?xf16>
            loom.semaphore_give %11 : memref<?xf16>
            %transposed_0 = linalg.transpose ins(%93 : tensor<?x?xf16>) outs(%40 : tensor<?x?xf16>) permutation = [1, 0] 
            loom.semaphore_give %52 : memref<?x?xf16>
            %94 = loom.subview %arg2[%59, %60, %61, %88] [1, 1, 1, %2] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x32x8x192xf16> to memref<?xf16, strided<[1], offset: ?>>
            loom.copy %94, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?xf16, strided<[1], offset: ?>> to memref<?xf16>
            %95 = loom.bufferize_to_tensor %50[%2] : memref<?xf16> -> tensor<?xf16>
            %96 = loom.broadcast ins(%95 : tensor<?xf16>) outs(%43 : tensor<?x?xf16>) dim(0) -> tensor<?x?xf16>
            loom.semaphore_give %50 : memref<?xf16>
            %97 = loom.broadcast ins(%92 : tensor<?xf16>) outs(%46 : tensor<?x?xf16>) dim(0) -> tensor<?x?xf16>
            loom.semaphore_give %48 : memref<?xf16>
            %98 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%90, %transposed_0, %97, %96 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>, tensor<?x?xf16>) outs(%40 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_1: f16, %in_2: f16, %in_3: f16, %out: f16):
              %105 = arith.subf %in_1, %in_2 : f16
              %106 = math.exp %105 : f16
              %107 = arith.mulf %in, %106 : f16
              %108 = arith.mulf %107, %in_3 : f16
              linalg.yield %108 : f16
            } -> tensor<?x?xf16>
            loom.semaphore_give %45 : memref<?x?xf16>
            loom.semaphore_give %42 : memref<?x?xf16>
            loom.semaphore_give %37 : memref<?x?xf16>
            %99 = arith.addi %88, %67 : index
            %100 = loom.subview %arg1[%59, %60, %99, %71] [1, 1, %2, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x32x1536x160xf16> to memref<?x?xf16, strided<[160, 1], offset: ?>>
            loom.copy %100, %55 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[160, 1], offset: ?>> to memref<?x?xf16>
            %101 = loom.bufferize_to_tensor %55[%2, %1] : memref<?x?xf16> -> tensor<?x?xf16>
            %102 = linalg.fill ins(%cst : f16) outs(%17 : tensor<?x?xf16>) -> tensor<?x?xf16>
            %103 = linalg.matmul ins(%98, %101 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%102 : tensor<?x?xf16>) -> tensor<?x?xf16>
            loom.semaphore_give %55 : memref<?x?xf16>
            loom.semaphore_give %39 : memref<?x?xf16>
            %104 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg14, %103 : tensor<?x?xf16>, tensor<?x?xf16>) outs(%26 : tensor<?x?xf16>) {
            ^bb0(%in: f16, %in_1: f16, %out: f16):
              %105 = arith.addf %in, %in_1 : f16
              linalg.yield %105 : f16
            } -> tensor<?x?xf16>
            loom.semaphore_give %16 : memref<?x?xf16>
            loom.semaphore_give %25 : memref<?x?xf16>
            scf.yield %104 : tensor<?x?xf16>
          }
          loom.semaphore_give %18 : memref<?x?xf16>
          %81 = loom.subview %arg5[%60] [1] [1], reuse : [seq = false, spat = false, temp = false] : memref<32xf16> to memref<f16, strided<[], offset: ?>>
          loom.copy %81, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<f16, strided<[], offset: ?>> to memref<f16>
          %82 = loom.bufferize_to_tensor %57[] : memref<f16> -> tensor<f16>
          %83 = loom.subview %arg1[%59, %60, %68, %71] [1, 1, %0, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x32x1536x160xf16> to memref<?x?xf16, strided<[160, 1], offset: ?>>
          loom.copy %83, %28 src_mem_space @mem_DRAM dst_mem_space @mem_L1, area : [1, 1] : memref<?x?xf16, strided<[160, 1], offset: ?>> to memref<?x?xf16>
          %84 = loom.bufferize_to_tensor %28[%0, %1] : memref<?x?xf16> -> tensor<?x?xf16>
          %85 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80, %84, %82 : tensor<?x?xf16>, tensor<?x?xf16>, tensor<f16>) outs(%31 : tensor<?x?xf16>) {
          ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
            %88 = arith.mulf %in_0, %in_1 : f16
            %89 = arith.addf %in, %88 : f16
            linalg.yield %89 : f16
          } -> tensor<?x?xf16>
          loom.semaphore_give %57 : memref<f16>
          loom.semaphore_give %28 : memref<?x?xf16>
          loom.semaphore_give %21 : memref<?x?xf16>
          %86 = loom.subview %arg7[%59, %60, %68, %71] [1, 1, %0, %1] [1, 1, 1, 1], reuse : [seq = false, spat = false, temp = false] : memref<2x32x1536x160xf16> to memref<?x?xf16, strided<[160, 1], offset: ?>>
          %87 = loom.bufferize_to_memref %85 : tensor<?x?xf16> -> memref<?x?xf16>
          loom.copy %87, %86 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[160, 1], offset: ?>>
          loom.semaphore_give %30 : memref<?x?xf16>
        }
      }
    }
    return
  }
}
