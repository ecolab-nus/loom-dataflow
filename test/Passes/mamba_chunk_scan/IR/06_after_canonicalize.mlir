module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
  %0 = adl.memory.bank "mem_DRAM_bank", {bsize = 8192 : i64, nblk = 196608 : i64}
  %1 = adl.spatial_dim "dim_dram_channel", 8
  %2 = adl.memory.array "mem_DRAM", [%1] of %0
  %3 = adl.resource.exclusive "res_L1_torus_h"
  %4 = adl.resource.exclusive "res_L1_torus_v"
  %5 = adl.memory.bank "mem_bank", {bsize = 16 : i64, nblk = 5856 : i64}
  %6 = adl.spatial_dim "dim_nbank", 16
  %7 = adl.memory.array "mem_L1", [%6] of %5
  %8 = adl.resource.exclusive "res_matrix_lane"
  %9 = adl.resource.exclusive "res_vector_lane"
  %10 = adl.processor.compute @proc_matrix_lane, [(%7, %7)], with [%8]
  %11 = adl.processor.compute @proc_vector_lane, [(%7, %7)], with [%9]
  %12 = adl.arch.compose "arch_core", arch[%10, %11], mem[%7]
  %13 = adl.spatial_dim "dim_x", 8
  %14 = adl.spatial_dim "dim_y", 8
  %15 = adl.arch.scale "arch_mesh", [%13, %14] of %12
  %16 = adl.processor.dmover @proc_dram_l1_mover, [(%2, %7), (%7, %2)], with [%3, %4]
  %17 = adl.processor.dmover @proc_dram_l1_bcst_v, [(%2, %7), (%7, %2)], with [%4]
  %18 = adl.processor.dmover @proc_dram_l1_bcst_h, [(%2, %7), (%7, %2)], with [%3]
  %19 = adl.arch.compose "arch_system", arch[%15, %16, %17, %18], mem[%2]
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c8 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg11, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg12, %c2 : index
                  %30 = arith.muli %arg8, %c4 : index
                  %31 = arith.addi %29, %30 : index
                  %32 = arith.addi %29, %c1 : index
                  %33 = arith.addi %32, %30 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %34 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %35 = loom.alloc [64] on @L1 : memref<64xf32>
                  %36 = loom.semaphore_take %35 : memref<64xf32> -> memref<64xf32>
                  %37 = loom.init_tensor %36[64] : memref<64xf32> -> tensor<64xf32>
                  %38 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%34 : tensor<64xf16>) outs(%37 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %99 = arith.extf %in : f16 to f32
                    linalg.yield %99 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %25 : memref<64xf16>
                  %39 = loom.alloc [64] on @L1 : memref<64xf32>
                  %40 = loom.semaphore_take %39 : memref<64xf32> -> memref<64xf32>
                  %41 = loom.init_tensor %40[64] : memref<64xf32> -> tensor<64xf32>
                  %42 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%38 : tensor<64xf32>) outs(%41 : tensor<64xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %99 = arith.truncf %cst_0 : f64 to f32
                    %100 = arith.mulf %in, %99 : f32
                    %101 = math.powf %cst, %100 : f32
                    linalg.yield %101 : f32
                  } -> tensor<64xf32>
                  %43 = arith.muli %arg12, %c1024 : index
                  %44 = arith.divui %21, %c16 : index
                  %45 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
                  %46 = loom.semaphore_take %45 : memref<64x16xf16> -> memref<64x16xf16>
                  %c0_2 = arith.constant 0 : index
                  %47 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 16 + d2 * 16 + d3)>(%arg11, %43, %44, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %48 = arith.addi %27, %c1 : index
                  loom.copy %reinterpret_cast_3, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%48, %33]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
                  %49 = loom.bufferize_to_tensor %46[64, 16] : memref<64x16xf16> -> tensor<64x16xf16>
                  %50 = arith.muli %arg10, %c32 : index
                  %51 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
                  %52 = loom.semaphore_take %51 : memref<32x16xf16> -> memref<32x16xf16>
                  %c0_4 = arith.constant 0 : index
                  %53 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 131072 + d1 * 16384 + d2 * 1024 + d3 * 16 + d4)>(%arg11, %22, %21, %50, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%53], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %54 = arith.addi %arg10, %29 : index
                  %55 = arith.addi %54, %30 : index
                  loom.copy %reinterpret_cast_5, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %55], LR : [%48, %55]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
                  %56 = loom.bufferize_to_tensor %52[32, 16] : memref<32x16xf16> -> tensor<32x16xf16>
                  %57 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
                  %58 = loom.semaphore_take %57 : memref<16x32xf16> -> memref<16x32xf16>
                  %59 = loom.init_tensor %58[16, 32] : memref<16x32xf16> -> tensor<16x32xf16>
                  %transposed = linalg.transpose ins(%56 : tensor<32x16xf16>) outs(%59 : tensor<16x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %52 : memref<32x16xf16>
                  %60 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
                  %61 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %62 = loom.init_tensor %61[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %63 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %64 = loom.init_tensor %63[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %65 = linalg.fill ins(%cst_1 : f32) outs(%62 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  %66 = linalg.matmul ins(%49, %transposed : tensor<64x16xf16>, tensor<16x32xf16>) outs(%65 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  loom.semaphore_give %58 : memref<16x32xf16>
                  loom.semaphore_give %46 : memref<64x16xf16>
                  %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %42 : tensor<64x32xf32>, tensor<64xf32>) outs(%64 : tensor<64x32xf32>) {
                  ^bb0(%in: f32, %in_9: f32, %out: f32):
                    %99 = arith.mulf %in, %in_9 : f32
                    linalg.yield %99 : f32
                  } -> tensor<64x32xf32>
                  loom.semaphore_give %61 : memref<64x32xf32>
                  loom.semaphore_give %40 : memref<64xf32>
                  %68 = arith.addi %20, %c1 : index
                  %69 = arith.muli %68, %c64 : index
                  %70 = arith.ceildivui %69, %c64 : index
                  %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
                  %73 = loom.init_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %74 = loom.alloc [64] on @L1 : memref<64xf16>
                  %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %76 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %77 = loom.alloc [64] on @L1 : memref<64xf32>
                  %78 = loom.semaphore_take %77 : memref<64xf32> -> memref<64xf32>
                  %79 = loom.init_tensor %78[64] : memref<64xf32> -> tensor<64xf32>
                  %80 = loom.alloc [64] on @L1 : memref<64xf32>
                  %81 = loom.semaphore_take %80 : memref<64xf32> -> memref<64xf32>
                  %82 = loom.init_tensor %81[64] : memref<64xf32> -> tensor<64xf32>
                  %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
                  %85 = scf.for %arg14 = %c0 to %70 step %c1 iter_args(%arg15 = %67) -> (tensor<64x32xf32>) {
                    %99 = arith.muli %arg14, %c64 : index
                    %100 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %44, %23, %99)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%100], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %101 = loom.bufferize_to_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %102 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%48, %33]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %103 = loom.bufferize_to_tensor %76[64] : memref<64xf16> -> tensor<64xf16>
                    %104 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%103 : tensor<64xf16>) outs(%79 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %76 : memref<64xf16>
                    %105 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%105], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%48, %33]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %106 = loom.bufferize_to_tensor %75[64] : memref<64xf16> -> tensor<64xf16>
                    %107 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%106 : tensor<64xf16>) outs(%82 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %75 : memref<64xf16>
                    %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %38, %104, %107 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%73 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f32, %in_14: f32, %in_15: f32, %out: f16):
                      %112 = arith.truncf %cst_0 : f64 to f32
                      %113 = arith.mulf %in_14, %112 : f32
                      %114 = arith.mulf %in_13, %112 : f32
                      %115 = arith.subf %114, %113 : f32
                      %116 = math.powf %cst, %115 : f32
                      %117 = arith.extf %in : f16 to f32
                      %118 = arith.mulf %117, %116 : f32
                      %119 = arith.mulf %118, %in_15 : f32
                      %120 = arith.truncf %119 : f32 to f16
                      linalg.yield %120 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %81 : memref<64xf32>
                    loom.semaphore_give %78 : memref<64xf32>
                    %109 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %43, %21, %50)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%109], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %55], LR : [%48, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                    %110 = loom.bufferize_to_tensor %84[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %111 = linalg.matmul ins(%108, %110 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf32>) -> tensor<64x32xf32>
                    loom.semaphore_give %84 : memref<64x32xf16>
                    loom.semaphore_give %72 : memref<64x64xf16>
                    scf.yield %111 : tensor<64x32xf32>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %36 : memref<64xf32>
                  %86 = loom.alloc [1] on @L1 : memref<f16>
                  %87 = loom.semaphore_take %86 : memref<f16> -> memref<f16>
                  %88 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%88], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                  %89 = arith.addi %30, %c3 : index
                  loom.copy %reinterpret_cast_6, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %30], LR : [%c7, %89]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %90 = loom.bufferize_to_tensor %87[] : memref<f16> -> tensor<f16>
                  %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
                  %93 = loom.init_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %43, %21, %50)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%94], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %55], LR : [%48, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                  %95 = loom.bufferize_to_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %95, %90 : tensor<64x32xf32>, tensor<64x32xf16>, tensor<f16>) outs(%93 : tensor<64x32xf16>) {
                  ^bb0(%in: f32, %in_9: f16, %in_10: f16, %out: f16):
                    %99 = arith.extf %in_10 : f16 to f32
                    %100 = arith.extf %in_9 : f16 to f32
                    %101 = arith.mulf %100, %99 : f32
                    %102 = arith.addf %in, %101 : f32
                    %103 = arith.truncf %102 : f32 to f16
                    linalg.yield %103 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %87 : memref<f16>
                  loom.semaphore_give %63 : memref<64x32xf32>
                  %97 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %43, %21, %50)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%97], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  %98 = loom.bufferize_to_memref %96 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %98, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%27, %55], LR : [%48, %55]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.semaphore_give %92 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c8 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg12, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg11, %c2 : index
                  %30 = arith.muli %arg8, %c4 : index
                  %31 = arith.addi %29, %30 : index
                  %32 = arith.addi %29, %c1 : index
                  %33 = arith.addi %32, %30 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %34 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %35 = loom.alloc [64] on @L1 : memref<64xf32>
                  %36 = loom.semaphore_take %35 : memref<64xf32> -> memref<64xf32>
                  %37 = loom.init_tensor %36[64] : memref<64xf32> -> tensor<64xf32>
                  %38 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%34 : tensor<64xf16>) outs(%37 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %99 = arith.extf %in : f16 to f32
                    linalg.yield %99 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %25 : memref<64xf16>
                  %39 = loom.alloc [64] on @L1 : memref<64xf32>
                  %40 = loom.semaphore_take %39 : memref<64xf32> -> memref<64xf32>
                  %41 = loom.init_tensor %40[64] : memref<64xf32> -> tensor<64xf32>
                  %42 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%38 : tensor<64xf32>) outs(%41 : tensor<64xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %99 = arith.truncf %cst_0 : f64 to f32
                    %100 = arith.mulf %in, %99 : f32
                    %101 = math.powf %cst, %100 : f32
                    linalg.yield %101 : f32
                  } -> tensor<64xf32>
                  %43 = arith.muli %arg12, %c1024 : index
                  %44 = arith.divui %21, %c16 : index
                  %45 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
                  %46 = loom.semaphore_take %45 : memref<64x16xf16> -> memref<64x16xf16>
                  %c0_2 = arith.constant 0 : index
                  %47 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 16 + d2 * 16 + d3)>(%arg11, %43, %44, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%47], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %48 = arith.addi %27, %c1 : index
                  loom.copy %reinterpret_cast_3, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%48, %33]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
                  %49 = loom.bufferize_to_tensor %46[64, 16] : memref<64x16xf16> -> tensor<64x16xf16>
                  %50 = arith.muli %arg10, %c32 : index
                  %51 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
                  %52 = loom.semaphore_take %51 : memref<32x16xf16> -> memref<32x16xf16>
                  %c0_4 = arith.constant 0 : index
                  %53 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 131072 + d1 * 16384 + d2 * 1024 + d3 * 16 + d4)>(%arg11, %22, %21, %50, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%53], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %54 = arith.addi %arg10, %29 : index
                  %55 = arith.addi %54, %30 : index
                  loom.copy %reinterpret_cast_5, %52 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %55], LR : [%48, %55]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
                  %56 = loom.bufferize_to_tensor %52[32, 16] : memref<32x16xf16> -> tensor<32x16xf16>
                  %57 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
                  %58 = loom.semaphore_take %57 : memref<16x32xf16> -> memref<16x32xf16>
                  %59 = loom.init_tensor %58[16, 32] : memref<16x32xf16> -> tensor<16x32xf16>
                  %transposed = linalg.transpose ins(%56 : tensor<32x16xf16>) outs(%59 : tensor<16x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %52 : memref<32x16xf16>
                  %60 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
                  %61 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %62 = loom.init_tensor %61[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %63 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %64 = loom.init_tensor %63[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %65 = linalg.fill ins(%cst_1 : f32) outs(%62 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  %66 = linalg.matmul ins(%49, %transposed : tensor<64x16xf16>, tensor<16x32xf16>) outs(%65 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  loom.semaphore_give %58 : memref<16x32xf16>
                  loom.semaphore_give %46 : memref<64x16xf16>
                  %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %42 : tensor<64x32xf32>, tensor<64xf32>) outs(%64 : tensor<64x32xf32>) {
                  ^bb0(%in: f32, %in_9: f32, %out: f32):
                    %99 = arith.mulf %in, %in_9 : f32
                    linalg.yield %99 : f32
                  } -> tensor<64x32xf32>
                  loom.semaphore_give %61 : memref<64x32xf32>
                  loom.semaphore_give %40 : memref<64xf32>
                  %68 = arith.addi %20, %c1 : index
                  %69 = arith.muli %68, %c64 : index
                  %70 = arith.ceildivui %69, %c64 : index
                  %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
                  %73 = loom.init_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %74 = loom.alloc [64] on @L1 : memref<64xf16>
                  %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %76 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %77 = loom.alloc [64] on @L1 : memref<64xf32>
                  %78 = loom.semaphore_take %77 : memref<64xf32> -> memref<64xf32>
                  %79 = loom.init_tensor %78[64] : memref<64xf32> -> tensor<64xf32>
                  %80 = loom.alloc [64] on @L1 : memref<64xf32>
                  %81 = loom.semaphore_take %80 : memref<64xf32> -> memref<64xf32>
                  %82 = loom.init_tensor %81[64] : memref<64xf32> -> tensor<64xf32>
                  %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
                  %85 = scf.for %arg14 = %c0 to %70 step %c1 iter_args(%arg15 = %67) -> (tensor<64x32xf32>) {
                    %99 = arith.muli %arg14, %c64 : index
                    %100 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %44, %23, %99)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%100], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%28, %31], LR : [%28, %33]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %101 = loom.bufferize_to_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %102 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%48, %33]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %103 = loom.bufferize_to_tensor %76[64] : memref<64xf16> -> tensor<64xf16>
                    %104 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%103 : tensor<64xf16>) outs(%79 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %76 : memref<64xf16>
                    %105 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%105], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%27, %31], LR : [%48, %33]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %106 = loom.bufferize_to_tensor %75[64] : memref<64xf16> -> tensor<64xf16>
                    %107 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%106 : tensor<64xf16>) outs(%82 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %75 : memref<64xf16>
                    %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %38, %104, %107 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%73 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f32, %in_14: f32, %in_15: f32, %out: f16):
                      %112 = arith.truncf %cst_0 : f64 to f32
                      %113 = arith.mulf %in_14, %112 : f32
                      %114 = arith.mulf %in_13, %112 : f32
                      %115 = arith.subf %114, %113 : f32
                      %116 = math.powf %cst, %115 : f32
                      %117 = arith.extf %in : f16 to f32
                      %118 = arith.mulf %117, %116 : f32
                      %119 = arith.mulf %118, %in_15 : f32
                      %120 = arith.truncf %119 : f32 to f16
                      linalg.yield %120 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %81 : memref<64xf32>
                    loom.semaphore_give %78 : memref<64xf32>
                    %109 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %43, %21, %50)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%109], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %55], LR : [%48, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                    %110 = loom.bufferize_to_tensor %84[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %111 = linalg.matmul ins(%108, %110 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf32>) -> tensor<64x32xf32>
                    loom.semaphore_give %84 : memref<64x32xf16>
                    loom.semaphore_give %72 : memref<64x64xf16>
                    scf.yield %111 : tensor<64x32xf32>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %36 : memref<64xf32>
                  %86 = loom.alloc [1] on @L1 : memref<f16>
                  %87 = loom.semaphore_take %86 : memref<f16> -> memref<f16>
                  %88 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%88], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                  %89 = arith.addi %30, %c3 : index
                  loom.copy %reinterpret_cast_6, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %30], LR : [%c7, %89]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %90 = loom.bufferize_to_tensor %87[] : memref<f16> -> tensor<f16>
                  %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
                  %93 = loom.init_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %43, %21, %50)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%94], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%27, %55], LR : [%48, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                  %95 = loom.bufferize_to_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %95, %90 : tensor<64x32xf32>, tensor<64x32xf16>, tensor<f16>) outs(%93 : tensor<64x32xf16>) {
                  ^bb0(%in: f32, %in_9: f16, %in_10: f16, %out: f16):
                    %99 = arith.extf %in_10 : f16 to f32
                    %100 = arith.extf %in_9 : f16 to f32
                    %101 = arith.mulf %100, %99 : f32
                    %102 = arith.addf %in, %101 : f32
                    %103 = arith.truncf %102 : f32 to f16
                    linalg.yield %103 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %87 : memref<f16>
                  loom.semaphore_give %63 : memref<64x32xf32>
                  %97 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %43, %21, %50)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%97], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  %98 = loom.bufferize_to_memref %96 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %98, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%27, %55], LR : [%48, %55]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.semaphore_give %92 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %20 = arith.muli %arg8, %c8 : index
                %21 = arith.muli %arg12, %c4 : index
                %22 = arith.muli %arg9, %c64 : index
                %23 = loom.alloc [64] on @L1 : memref<64xf16>
                %24 = loom.semaphore_take %23 : memref<64xf16> -> memref<64xf16>
                %25 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %22)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                %26 = arith.muli %arg11, %c4 : index
                %27 = arith.addi %arg9, %26 : index
                %28 = arith.muli %arg12, %c2 : index
                %29 = arith.muli %arg8, %c4 : index
                %30 = arith.addi %28, %29 : index
                %31 = arith.addi %28, %c1 : index
                %32 = arith.addi %31, %29 : index
                loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%27, %30], LR : [%27, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                %33 = loom.bufferize_to_tensor %24[64] : memref<64xf16> -> tensor<64xf16>
                %34 = loom.alloc [64] on @L1 : memref<64xf32>
                %35 = loom.semaphore_take %34 : memref<64xf32> -> memref<64xf32>
                %36 = loom.init_tensor %35[64] : memref<64xf32> -> tensor<64xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%33 : tensor<64xf16>) outs(%36 : tensor<64xf32>) {
                ^bb0(%in: f16, %out: f32):
                  %98 = arith.extf %in : f16 to f32
                  linalg.yield %98 : f32
                } -> tensor<64xf32>
                loom.semaphore_give %24 : memref<64xf16>
                %38 = loom.alloc [64] on @L1 : memref<64xf32>
                %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
                %40 = loom.init_tensor %39[64] : memref<64xf32> -> tensor<64xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%37 : tensor<64xf32>) outs(%40 : tensor<64xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %98 = arith.truncf %cst_0 : f64 to f32
                  %99 = arith.mulf %in, %98 : f32
                  %100 = math.powf %cst, %99 : f32
                  linalg.yield %100 : f32
                } -> tensor<64xf32>
                %42 = arith.muli %arg12, %c1024 : index
                %43 = arith.divui %20, %c16 : index
                %44 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
                %45 = loom.semaphore_take %44 : memref<64x16xf16> -> memref<64x16xf16>
                %c0_2 = arith.constant 0 : index
                %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 16 + d2 * 16 + d3)>(%arg11, %42, %43, %c0_2)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                %47 = arith.addi %26, %c3 : index
                loom.copy %reinterpret_cast_3, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%47, %32]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
                %48 = loom.bufferize_to_tensor %45[64, 16] : memref<64x16xf16> -> tensor<64x16xf16>
                %49 = arith.muli %arg10, %c32 : index
                %50 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
                %51 = loom.semaphore_take %50 : memref<32x16xf16> -> memref<32x16xf16>
                %c0_4 = arith.constant 0 : index
                %52 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 131072 + d1 * 16384 + d2 * 1024 + d3 * 16 + d4)>(%arg11, %21, %20, %49, %c0_4)
                %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                %53 = arith.addi %arg10, %28 : index
                %54 = arith.addi %53, %29 : index
                loom.copy %reinterpret_cast_5, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %54], LR : [%47, %54]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
                %55 = loom.bufferize_to_tensor %51[32, 16] : memref<32x16xf16> -> tensor<32x16xf16>
                %56 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
                %57 = loom.semaphore_take %56 : memref<16x32xf16> -> memref<16x32xf16>
                %58 = loom.init_tensor %57[16, 32] : memref<16x32xf16> -> tensor<16x32xf16>
                %transposed = linalg.transpose ins(%55 : tensor<32x16xf16>) outs(%58 : tensor<16x32xf16>) permutation = [1, 0] 
                loom.semaphore_give %51 : memref<32x16xf16>
                %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
                %60 = loom.semaphore_take %59 : memref<64x32xf32> -> memref<64x32xf32>
                %61 = loom.init_tensor %60[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                %62 = loom.semaphore_take %59 : memref<64x32xf32> -> memref<64x32xf32>
                %63 = loom.init_tensor %62[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                %64 = linalg.fill ins(%cst_1 : f32) outs(%61 : tensor<64x32xf32>) -> tensor<64x32xf32>
                %65 = linalg.matmul ins(%48, %transposed : tensor<64x16xf16>, tensor<16x32xf16>) outs(%64 : tensor<64x32xf32>) -> tensor<64x32xf32>
                loom.semaphore_give %57 : memref<16x32xf16>
                loom.semaphore_give %45 : memref<64x16xf16>
                %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %41 : tensor<64x32xf32>, tensor<64xf32>) outs(%63 : tensor<64x32xf32>) {
                ^bb0(%in: f32, %in_9: f32, %out: f32):
                  %98 = arith.mulf %in, %in_9 : f32
                  linalg.yield %98 : f32
                } -> tensor<64x32xf32>
                loom.semaphore_give %60 : memref<64x32xf32>
                loom.semaphore_give %39 : memref<64xf32>
                %67 = arith.addi %arg9, %c1 : index
                %68 = arith.muli %67, %c64 : index
                %69 = arith.ceildivui %68, %c64 : index
                %70 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %71 = loom.semaphore_take %70 : memref<64x64xf16> -> memref<64x64xf16>
                %72 = loom.init_tensor %71[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %73 = loom.alloc [64] on @L1 : memref<64xf16>
                %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                %76 = loom.alloc [64] on @L1 : memref<64xf32>
                %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
                %78 = loom.init_tensor %77[64] : memref<64xf32> -> tensor<64xf32>
                %79 = loom.alloc [64] on @L1 : memref<64xf32>
                %80 = loom.semaphore_take %79 : memref<64xf32> -> memref<64xf32>
                %81 = loom.init_tensor %80[64] : memref<64xf32> -> tensor<64xf32>
                %82 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %83 = loom.semaphore_take %82 : memref<64x32xf16> -> memref<64x32xf16>
                %84 = scf.for %arg13 = %c0 to %69 step %c1 iter_args(%arg14 = %66) -> (tensor<64x32xf32>) {
                  %98 = arith.muli %arg13, %c64 : index
                  %99 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %21, %43, %22, %98)
                  %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%99], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %reinterpret_cast_9, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%27, %30], LR : [%27, %32]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                  %100 = loom.bufferize_to_tensor %71[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %101 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %98)
                  %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%101], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  loom.copy %reinterpret_cast_10, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%47, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %102 = loom.bufferize_to_tensor %75[64] : memref<64xf16> -> tensor<64xf16>
                  %103 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%102 : tensor<64xf16>) outs(%78 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %111 = arith.extf %in : f16 to f32
                    linalg.yield %111 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %75 : memref<64xf16>
                  %104 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %98)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%104], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  loom.copy %reinterpret_cast_11, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%47, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %105 = loom.bufferize_to_tensor %74[64] : memref<64xf16> -> tensor<64xf16>
                  %106 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%105 : tensor<64xf16>) outs(%81 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %111 = arith.extf %in : f16 to f32
                    linalg.yield %111 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %74 : memref<64xf16>
                  %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%100, %37, %103, %106 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%72 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %in_13: f32, %in_14: f32, %in_15: f32, %out: f16):
                    %111 = arith.truncf %cst_0 : f64 to f32
                    %112 = arith.mulf %in_14, %111 : f32
                    %113 = arith.mulf %in_13, %111 : f32
                    %114 = arith.subf %113, %112 : f32
                    %115 = math.powf %cst, %114 : f32
                    %116 = arith.extf %in : f16 to f32
                    %117 = arith.mulf %116, %115 : f32
                    %118 = arith.mulf %117, %in_15 : f32
                    %119 = arith.truncf %118 : f32 to f16
                    linalg.yield %119 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %80 : memref<64xf32>
                  loom.semaphore_give %77 : memref<64xf32>
                  %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %20, %49)
                  %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.copy %reinterpret_cast_12, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %54], LR : [%47, %54]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                  %109 = loom.bufferize_to_tensor %83[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %110 = linalg.matmul ins(%107, %109 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%arg14 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  loom.semaphore_give %83 : memref<64x32xf16>
                  loom.semaphore_give %71 : memref<64x64xf16>
                  scf.yield %110 : tensor<64x32xf32>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %35 : memref<64xf32>
                %85 = loom.alloc [1] on @L1 : memref<f16>
                %86 = loom.semaphore_take %85 : memref<f16> -> memref<f16>
                %87 = affine.apply affine_map<(d0) -> (d0)>(%20)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%87], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                %88 = arith.addi %29, %c3 : index
                loom.copy %reinterpret_cast_6, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %29], LR : [%c7, %88]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                %89 = loom.bufferize_to_tensor %86[] : memref<f16> -> tensor<f16>
                %90 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %91 = loom.semaphore_take %90 : memref<64x32xf16> -> memref<64x32xf16>
                %92 = loom.init_tensor %91[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %93 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %20, %49)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%93], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %54], LR : [%47, %54]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                %94 = loom.bufferize_to_tensor %91[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%84, %94, %89 : tensor<64x32xf32>, tensor<64x32xf16>, tensor<f16>) outs(%92 : tensor<64x32xf16>) {
                ^bb0(%in: f32, %in_9: f16, %in_10: f16, %out: f16):
                  %98 = arith.extf %in_10 : f16 to f32
                  %99 = arith.extf %in_9 : f16 to f32
                  %100 = arith.mulf %99, %98 : f32
                  %101 = arith.addf %in, %100 : f32
                  %102 = arith.truncf %101 : f32 to f16
                  linalg.yield %102 : f16
                } -> tensor<64x32xf16>
                loom.semaphore_give %86 : memref<f16>
                loom.semaphore_give %62 : memref<64x32xf32>
                %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %20, %49)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%96], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                %97 = loom.bufferize_to_memref %95 : tensor<64x32xf16> -> memref<64x32xf16>
                loom.copy %97, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%26, %54], LR : [%47, %54]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                loom.semaphore_give %91 : memref<64x32xf16>
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_y_level0_bc2_dim_x_level0_bc4_dim_x_level1_bc8_dim_y_level1_bc4_dim_x_level0_bc4_dim_x_level0_bc4__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (4) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                %20 = arith.muli %arg8, %c8 : index
                %21 = arith.muli %arg12, %c4 : index
                %22 = arith.muli %arg9, %c64 : index
                %23 = loom.alloc [64] on @L1 : memref<64xf16>
                %24 = loom.semaphore_take %23 : memref<64xf16> -> memref<64xf16>
                %25 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %22)
                %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%25], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                %26 = arith.muli %arg12, %c4 : index
                %27 = arith.addi %arg9, %26 : index
                %28 = arith.muli %arg11, %c2 : index
                %29 = arith.muli %arg8, %c4 : index
                %30 = arith.addi %28, %29 : index
                %31 = arith.addi %28, %c1 : index
                %32 = arith.addi %31, %29 : index
                loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%27, %30], LR : [%27, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                %33 = loom.bufferize_to_tensor %24[64] : memref<64xf16> -> tensor<64xf16>
                %34 = loom.alloc [64] on @L1 : memref<64xf32>
                %35 = loom.semaphore_take %34 : memref<64xf32> -> memref<64xf32>
                %36 = loom.init_tensor %35[64] : memref<64xf32> -> tensor<64xf32>
                %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%33 : tensor<64xf16>) outs(%36 : tensor<64xf32>) {
                ^bb0(%in: f16, %out: f32):
                  %98 = arith.extf %in : f16 to f32
                  linalg.yield %98 : f32
                } -> tensor<64xf32>
                loom.semaphore_give %24 : memref<64xf16>
                %38 = loom.alloc [64] on @L1 : memref<64xf32>
                %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
                %40 = loom.init_tensor %39[64] : memref<64xf32> -> tensor<64xf32>
                %41 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%37 : tensor<64xf32>) outs(%40 : tensor<64xf32>) {
                ^bb0(%in: f32, %out: f32):
                  %98 = arith.truncf %cst_0 : f64 to f32
                  %99 = arith.mulf %in, %98 : f32
                  %100 = math.powf %cst, %99 : f32
                  linalg.yield %100 : f32
                } -> tensor<64xf32>
                %42 = arith.muli %arg12, %c1024 : index
                %43 = arith.divui %20, %c16 : index
                %44 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
                %45 = loom.semaphore_take %44 : memref<64x16xf16> -> memref<64x16xf16>
                %c0_2 = arith.constant 0 : index
                %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 16 + d2 * 16 + d3)>(%arg11, %42, %43, %c0_2)
                %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                %47 = arith.addi %26, %c3 : index
                loom.copy %reinterpret_cast_3, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%47, %32]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
                %48 = loom.bufferize_to_tensor %45[64, 16] : memref<64x16xf16> -> tensor<64x16xf16>
                %49 = arith.muli %arg10, %c32 : index
                %50 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
                %51 = loom.semaphore_take %50 : memref<32x16xf16> -> memref<32x16xf16>
                %c0_4 = arith.constant 0 : index
                %52 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 131072 + d1 * 16384 + d2 * 1024 + d3 * 16 + d4)>(%arg11, %21, %20, %49, %c0_4)
                %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%52], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                %53 = arith.addi %arg10, %28 : index
                %54 = arith.addi %53, %29 : index
                loom.copy %reinterpret_cast_5, %51 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %54], LR : [%47, %54]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
                %55 = loom.bufferize_to_tensor %51[32, 16] : memref<32x16xf16> -> tensor<32x16xf16>
                %56 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
                %57 = loom.semaphore_take %56 : memref<16x32xf16> -> memref<16x32xf16>
                %58 = loom.init_tensor %57[16, 32] : memref<16x32xf16> -> tensor<16x32xf16>
                %transposed = linalg.transpose ins(%55 : tensor<32x16xf16>) outs(%58 : tensor<16x32xf16>) permutation = [1, 0] 
                loom.semaphore_give %51 : memref<32x16xf16>
                %59 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
                %60 = loom.semaphore_take %59 : memref<64x32xf32> -> memref<64x32xf32>
                %61 = loom.init_tensor %60[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                %62 = loom.semaphore_take %59 : memref<64x32xf32> -> memref<64x32xf32>
                %63 = loom.init_tensor %62[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                %64 = linalg.fill ins(%cst_1 : f32) outs(%61 : tensor<64x32xf32>) -> tensor<64x32xf32>
                %65 = linalg.matmul ins(%48, %transposed : tensor<64x16xf16>, tensor<16x32xf16>) outs(%64 : tensor<64x32xf32>) -> tensor<64x32xf32>
                loom.semaphore_give %57 : memref<16x32xf16>
                loom.semaphore_give %45 : memref<64x16xf16>
                %66 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %41 : tensor<64x32xf32>, tensor<64xf32>) outs(%63 : tensor<64x32xf32>) {
                ^bb0(%in: f32, %in_9: f32, %out: f32):
                  %98 = arith.mulf %in, %in_9 : f32
                  linalg.yield %98 : f32
                } -> tensor<64x32xf32>
                loom.semaphore_give %60 : memref<64x32xf32>
                loom.semaphore_give %39 : memref<64xf32>
                %67 = arith.addi %arg9, %c1 : index
                %68 = arith.muli %67, %c64 : index
                %69 = arith.ceildivui %68, %c64 : index
                %70 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                %71 = loom.semaphore_take %70 : memref<64x64xf16> -> memref<64x64xf16>
                %72 = loom.init_tensor %71[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                %73 = loom.alloc [64] on @L1 : memref<64xf16>
                %74 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                %75 = loom.semaphore_take %73 : memref<64xf16> -> memref<64xf16>
                %76 = loom.alloc [64] on @L1 : memref<64xf32>
                %77 = loom.semaphore_take %76 : memref<64xf32> -> memref<64xf32>
                %78 = loom.init_tensor %77[64] : memref<64xf32> -> tensor<64xf32>
                %79 = loom.alloc [64] on @L1 : memref<64xf32>
                %80 = loom.semaphore_take %79 : memref<64xf32> -> memref<64xf32>
                %81 = loom.init_tensor %80[64] : memref<64xf32> -> tensor<64xf32>
                %82 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %83 = loom.semaphore_take %82 : memref<64x32xf16> -> memref<64x32xf16>
                %84 = scf.for %arg13 = %c0 to %69 step %c1 iter_args(%arg14 = %66) -> (tensor<64x32xf32>) {
                  %98 = arith.muli %arg13, %c64 : index
                  %99 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %21, %43, %22, %98)
                  %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%99], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                  loom.copy %reinterpret_cast_9, %71 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%27, %30], LR : [%27, %32]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                  %100 = loom.bufferize_to_tensor %71[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %101 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %98)
                  %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%101], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  loom.copy %reinterpret_cast_10, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%47, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %102 = loom.bufferize_to_tensor %75[64] : memref<64xf16> -> tensor<64xf16>
                  %103 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%102 : tensor<64xf16>) outs(%78 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %111 = arith.extf %in : f16 to f32
                    linalg.yield %111 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %75 : memref<64xf16>
                  %104 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %20, %21, %98)
                  %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%104], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  loom.copy %reinterpret_cast_11, %74 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 2] region : (UL : [%26, %30], LR : [%47, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %105 = loom.bufferize_to_tensor %74[64] : memref<64xf16> -> tensor<64xf16>
                  %106 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%105 : tensor<64xf16>) outs(%81 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %111 = arith.extf %in : f16 to f32
                    linalg.yield %111 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %74 : memref<64xf16>
                  %107 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%100, %37, %103, %106 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%72 : tensor<64x64xf16>) {
                  ^bb0(%in: f16, %in_13: f32, %in_14: f32, %in_15: f32, %out: f16):
                    %111 = arith.truncf %cst_0 : f64 to f32
                    %112 = arith.mulf %in_14, %111 : f32
                    %113 = arith.mulf %in_13, %111 : f32
                    %114 = arith.subf %113, %112 : f32
                    %115 = math.powf %cst, %114 : f32
                    %116 = arith.extf %in : f16 to f32
                    %117 = arith.mulf %116, %115 : f32
                    %118 = arith.mulf %117, %in_15 : f32
                    %119 = arith.truncf %118 : f32 to f16
                    linalg.yield %119 : f16
                  } -> tensor<64x64xf16>
                  loom.semaphore_give %80 : memref<64xf32>
                  loom.semaphore_give %77 : memref<64xf32>
                  %108 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %20, %49)
                  %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%108], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.copy %reinterpret_cast_12, %83 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %54], LR : [%47, %54]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                  %109 = loom.bufferize_to_tensor %83[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %110 = linalg.matmul ins(%107, %109 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%arg14 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  loom.semaphore_give %83 : memref<64x32xf16>
                  loom.semaphore_give %71 : memref<64x64xf16>
                  scf.yield %110 : tensor<64x32xf32>
                } {loom.iter_type = #loom.iter_type<sequential>}
                loom.semaphore_give %35 : memref<64xf32>
                %85 = loom.alloc [1] on @L1 : memref<f16>
                %86 = loom.semaphore_take %85 : memref<f16> -> memref<f16>
                %87 = affine.apply affine_map<(d0) -> (d0)>(%20)
                %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%87], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                %88 = arith.addi %29, %c3 : index
                loom.copy %reinterpret_cast_6, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %29], LR : [%c7, %88]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                %89 = loom.bufferize_to_tensor %86[] : memref<f16> -> tensor<f16>
                %90 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                %91 = loom.semaphore_take %90 : memref<64x32xf16> -> memref<64x32xf16>
                %92 = loom.init_tensor %91[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %93 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %20, %49)
                %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%93], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                loom.copy %reinterpret_cast_7, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%26, %54], LR : [%47, %54]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                %94 = loom.bufferize_to_tensor %91[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                %95 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%84, %94, %89 : tensor<64x32xf32>, tensor<64x32xf16>, tensor<f16>) outs(%92 : tensor<64x32xf16>) {
                ^bb0(%in: f32, %in_9: f16, %in_10: f16, %out: f16):
                  %98 = arith.extf %in_10 : f16 to f32
                  %99 = arith.extf %in_9 : f16 to f32
                  %100 = arith.mulf %99, %98 : f32
                  %101 = arith.addf %in, %100 : f32
                  %102 = arith.truncf %101 : f32 to f16
                  linalg.yield %102 : f16
                } -> tensor<64x32xf16>
                loom.semaphore_give %86 : memref<f16>
                loom.semaphore_give %62 : memref<64x32xf32>
                %96 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %20, %49)
                %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%96], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                %97 = loom.bufferize_to_memref %95 : tensor<64x32xf16> -> memref<64x32xf16>
                loom.copy %97, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [4, 1] region : (UL : [%26, %54], LR : [%47, %54]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                loom.semaphore_give %91 : memref<64x32xf16>
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_y}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (4) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c8 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg11, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg8, %c4 : index
                  %30 = arith.addi %28, %29 : index
                  %31 = arith.muli %arg12, %c2 : index
                  %32 = arith.addi %31, %c1 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %31], LR : [%30, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %33 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %34 = loom.alloc [64] on @L1 : memref<64xf32>
                  %35 = loom.semaphore_take %34 : memref<64xf32> -> memref<64xf32>
                  %36 = loom.init_tensor %35[64] : memref<64xf32> -> tensor<64xf32>
                  %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%33 : tensor<64xf16>) outs(%36 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %99 = arith.extf %in : f16 to f32
                    linalg.yield %99 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %25 : memref<64xf16>
                  %38 = loom.alloc [64] on @L1 : memref<64xf32>
                  %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
                  %40 = loom.init_tensor %39[64] : memref<64xf32> -> tensor<64xf32>
                  %41 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%37 : tensor<64xf32>) outs(%40 : tensor<64xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %99 = arith.truncf %cst_0 : f64 to f32
                    %100 = arith.mulf %in, %99 : f32
                    %101 = math.powf %cst, %100 : f32
                    linalg.yield %101 : f32
                  } -> tensor<64xf32>
                  %42 = arith.muli %arg12, %c1024 : index
                  %43 = arith.divui %21, %c16 : index
                  %44 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
                  %45 = loom.semaphore_take %44 : memref<64x16xf16> -> memref<64x16xf16>
                  %c0_2 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 16 + d2 * 16 + d3)>(%arg11, %42, %43, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %47 = arith.addi %27, %29 : index
                  %48 = arith.addi %27, %c1 : index
                  %49 = arith.addi %48, %29 : index
                  loom.copy %reinterpret_cast_3, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
                  %50 = loom.bufferize_to_tensor %45[64, 16] : memref<64x16xf16> -> tensor<64x16xf16>
                  %51 = arith.muli %arg10, %c32 : index
                  %52 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
                  %53 = loom.semaphore_take %52 : memref<32x16xf16> -> memref<32x16xf16>
                  %c0_4 = arith.constant 0 : index
                  %54 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 131072 + d1 * 16384 + d2 * 1024 + d3 * 16 + d4)>(%arg11, %22, %21, %51, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %55 = arith.addi %arg10, %31 : index
                  loom.copy %reinterpret_cast_5, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
                  %56 = loom.bufferize_to_tensor %53[32, 16] : memref<32x16xf16> -> tensor<32x16xf16>
                  %57 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
                  %58 = loom.semaphore_take %57 : memref<16x32xf16> -> memref<16x32xf16>
                  %59 = loom.init_tensor %58[16, 32] : memref<16x32xf16> -> tensor<16x32xf16>
                  %transposed = linalg.transpose ins(%56 : tensor<32x16xf16>) outs(%59 : tensor<16x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %53 : memref<32x16xf16>
                  %60 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
                  %61 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %62 = loom.init_tensor %61[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %63 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %64 = loom.init_tensor %63[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %65 = linalg.fill ins(%cst_1 : f32) outs(%62 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  %66 = linalg.matmul ins(%50, %transposed : tensor<64x16xf16>, tensor<16x32xf16>) outs(%65 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  loom.semaphore_give %58 : memref<16x32xf16>
                  loom.semaphore_give %45 : memref<64x16xf16>
                  %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %41 : tensor<64x32xf32>, tensor<64xf32>) outs(%64 : tensor<64x32xf32>) {
                  ^bb0(%in: f32, %in_9: f32, %out: f32):
                    %99 = arith.mulf %in, %in_9 : f32
                    linalg.yield %99 : f32
                  } -> tensor<64x32xf32>
                  loom.semaphore_give %61 : memref<64x32xf32>
                  loom.semaphore_give %39 : memref<64xf32>
                  %68 = arith.addi %20, %c1 : index
                  %69 = arith.muli %68, %c64 : index
                  %70 = arith.ceildivui %69, %c64 : index
                  %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
                  %73 = loom.init_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %74 = loom.alloc [64] on @L1 : memref<64xf16>
                  %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %76 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %77 = loom.alloc [64] on @L1 : memref<64xf32>
                  %78 = loom.semaphore_take %77 : memref<64xf32> -> memref<64xf32>
                  %79 = loom.init_tensor %78[64] : memref<64xf32> -> tensor<64xf32>
                  %80 = loom.alloc [64] on @L1 : memref<64xf32>
                  %81 = loom.semaphore_take %80 : memref<64xf32> -> memref<64xf32>
                  %82 = loom.init_tensor %81[64] : memref<64xf32> -> tensor<64xf32>
                  %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
                  %85 = scf.for %arg14 = %c0 to %70 step %c1 iter_args(%arg15 = %67) -> (tensor<64x32xf32>) {
                    %99 = arith.muli %arg14, %c64 : index
                    %100 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %43, %23, %99)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%100], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %31], LR : [%30, %32]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %101 = loom.bufferize_to_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %102 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %103 = loom.bufferize_to_tensor %76[64] : memref<64xf16> -> tensor<64xf16>
                    %104 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%103 : tensor<64xf16>) outs(%79 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %76 : memref<64xf16>
                    %105 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%105], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %106 = loom.bufferize_to_tensor %75[64] : memref<64xf16> -> tensor<64xf16>
                    %107 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%106 : tensor<64xf16>) outs(%82 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %75 : memref<64xf16>
                    %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %37, %104, %107 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%73 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f32, %in_14: f32, %in_15: f32, %out: f16):
                      %112 = arith.truncf %cst_0 : f64 to f32
                      %113 = arith.mulf %in_14, %112 : f32
                      %114 = arith.mulf %in_13, %112 : f32
                      %115 = arith.subf %114, %113 : f32
                      %116 = math.powf %cst, %115 : f32
                      %117 = arith.extf %in : f16 to f32
                      %118 = arith.mulf %117, %116 : f32
                      %119 = arith.mulf %118, %in_15 : f32
                      %120 = arith.truncf %119 : f32 to f16
                      linalg.yield %120 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %81 : memref<64xf32>
                    loom.semaphore_give %78 : memref<64xf32>
                    %109 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%109], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                    %110 = loom.bufferize_to_tensor %84[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %111 = linalg.matmul ins(%108, %110 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf32>) -> tensor<64x32xf32>
                    loom.semaphore_give %84 : memref<64x32xf16>
                    loom.semaphore_give %72 : memref<64x64xf16>
                    scf.yield %111 : tensor<64x32xf32>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %35 : memref<64xf32>
                  %86 = loom.alloc [1] on @L1 : memref<f16>
                  %87 = loom.semaphore_take %86 : memref<f16> -> memref<f16>
                  %88 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%88], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                  %89 = arith.addi %29, %c3 : index
                  loom.copy %reinterpret_cast_6, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%29, %c0], LR : [%89, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %90 = loom.bufferize_to_tensor %87[] : memref<f16> -> tensor<f16>
                  %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
                  %93 = loom.init_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%94], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                  %95 = loom.bufferize_to_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %95, %90 : tensor<64x32xf32>, tensor<64x32xf16>, tensor<f16>) outs(%93 : tensor<64x32xf16>) {
                  ^bb0(%in: f32, %in_9: f16, %in_10: f16, %out: f16):
                    %99 = arith.extf %in_10 : f16 to f32
                    %100 = arith.extf %in_9 : f16 to f32
                    %101 = arith.mulf %100, %99 : f32
                    %102 = arith.addf %in, %101 : f32
                    %103 = arith.truncf %102 : f32 to f16
                    linalg.yield %103 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %87 : memref<f16>
                  loom.semaphore_give %63 : memref<64x32xf32>
                  %97 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%97], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  %98 = loom.bufferize_to_memref %96 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %98, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.semaphore_give %92 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (2) {
            affine.parallel (%arg11) = (0) to (4) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c8 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg12, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg8, %c4 : index
                  %30 = arith.addi %28, %29 : index
                  %31 = arith.muli %arg11, %c2 : index
                  %32 = arith.addi %31, %c1 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %31], LR : [%30, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %33 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %34 = loom.alloc [64] on @L1 : memref<64xf32>
                  %35 = loom.semaphore_take %34 : memref<64xf32> -> memref<64xf32>
                  %36 = loom.init_tensor %35[64] : memref<64xf32> -> tensor<64xf32>
                  %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%33 : tensor<64xf16>) outs(%36 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %99 = arith.extf %in : f16 to f32
                    linalg.yield %99 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %25 : memref<64xf16>
                  %38 = loom.alloc [64] on @L1 : memref<64xf32>
                  %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
                  %40 = loom.init_tensor %39[64] : memref<64xf32> -> tensor<64xf32>
                  %41 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%37 : tensor<64xf32>) outs(%40 : tensor<64xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %99 = arith.truncf %cst_0 : f64 to f32
                    %100 = arith.mulf %in, %99 : f32
                    %101 = math.powf %cst, %100 : f32
                    linalg.yield %101 : f32
                  } -> tensor<64xf32>
                  %42 = arith.muli %arg12, %c1024 : index
                  %43 = arith.divui %21, %c16 : index
                  %44 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
                  %45 = loom.semaphore_take %44 : memref<64x16xf16> -> memref<64x16xf16>
                  %c0_2 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 16 + d2 * 16 + d3)>(%arg11, %42, %43, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %47 = arith.addi %27, %29 : index
                  %48 = arith.addi %27, %c1 : index
                  %49 = arith.addi %48, %29 : index
                  loom.copy %reinterpret_cast_3, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
                  %50 = loom.bufferize_to_tensor %45[64, 16] : memref<64x16xf16> -> tensor<64x16xf16>
                  %51 = arith.muli %arg10, %c32 : index
                  %52 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
                  %53 = loom.semaphore_take %52 : memref<32x16xf16> -> memref<32x16xf16>
                  %c0_4 = arith.constant 0 : index
                  %54 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 131072 + d1 * 16384 + d2 * 1024 + d3 * 16 + d4)>(%arg11, %22, %21, %51, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %55 = arith.addi %arg10, %31 : index
                  loom.copy %reinterpret_cast_5, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
                  %56 = loom.bufferize_to_tensor %53[32, 16] : memref<32x16xf16> -> tensor<32x16xf16>
                  %57 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
                  %58 = loom.semaphore_take %57 : memref<16x32xf16> -> memref<16x32xf16>
                  %59 = loom.init_tensor %58[16, 32] : memref<16x32xf16> -> tensor<16x32xf16>
                  %transposed = linalg.transpose ins(%56 : tensor<32x16xf16>) outs(%59 : tensor<16x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %53 : memref<32x16xf16>
                  %60 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
                  %61 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %62 = loom.init_tensor %61[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %63 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %64 = loom.init_tensor %63[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %65 = linalg.fill ins(%cst_1 : f32) outs(%62 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  %66 = linalg.matmul ins(%50, %transposed : tensor<64x16xf16>, tensor<16x32xf16>) outs(%65 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  loom.semaphore_give %58 : memref<16x32xf16>
                  loom.semaphore_give %45 : memref<64x16xf16>
                  %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %41 : tensor<64x32xf32>, tensor<64xf32>) outs(%64 : tensor<64x32xf32>) {
                  ^bb0(%in: f32, %in_9: f32, %out: f32):
                    %99 = arith.mulf %in, %in_9 : f32
                    linalg.yield %99 : f32
                  } -> tensor<64x32xf32>
                  loom.semaphore_give %61 : memref<64x32xf32>
                  loom.semaphore_give %39 : memref<64xf32>
                  %68 = arith.addi %20, %c1 : index
                  %69 = arith.muli %68, %c64 : index
                  %70 = arith.ceildivui %69, %c64 : index
                  %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
                  %73 = loom.init_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %74 = loom.alloc [64] on @L1 : memref<64xf16>
                  %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %76 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %77 = loom.alloc [64] on @L1 : memref<64xf32>
                  %78 = loom.semaphore_take %77 : memref<64xf32> -> memref<64xf32>
                  %79 = loom.init_tensor %78[64] : memref<64xf32> -> tensor<64xf32>
                  %80 = loom.alloc [64] on @L1 : memref<64xf32>
                  %81 = loom.semaphore_take %80 : memref<64xf32> -> memref<64xf32>
                  %82 = loom.init_tensor %81[64] : memref<64xf32> -> tensor<64xf32>
                  %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
                  %85 = scf.for %arg14 = %c0 to %70 step %c1 iter_args(%arg15 = %67) -> (tensor<64x32xf32>) {
                    %99 = arith.muli %arg14, %c64 : index
                    %100 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %43, %23, %99)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%100], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%30, %31], LR : [%30, %32]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %101 = loom.bufferize_to_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %102 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %103 = loom.bufferize_to_tensor %76[64] : memref<64xf16> -> tensor<64xf16>
                    %104 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%103 : tensor<64xf16>) outs(%79 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %76 : memref<64xf16>
                    %105 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%105], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 2] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %106 = loom.bufferize_to_tensor %75[64] : memref<64xf16> -> tensor<64xf16>
                    %107 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%106 : tensor<64xf16>) outs(%82 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %75 : memref<64xf16>
                    %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %37, %104, %107 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%73 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f32, %in_14: f32, %in_15: f32, %out: f16):
                      %112 = arith.truncf %cst_0 : f64 to f32
                      %113 = arith.mulf %in_14, %112 : f32
                      %114 = arith.mulf %in_13, %112 : f32
                      %115 = arith.subf %114, %113 : f32
                      %116 = math.powf %cst, %115 : f32
                      %117 = arith.extf %in : f16 to f32
                      %118 = arith.mulf %117, %116 : f32
                      %119 = arith.mulf %118, %in_15 : f32
                      %120 = arith.truncf %119 : f32 to f16
                      linalg.yield %120 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %81 : memref<64xf32>
                    loom.semaphore_give %78 : memref<64xf32>
                    %109 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%109], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                    %110 = loom.bufferize_to_tensor %84[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %111 = linalg.matmul ins(%108, %110 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf32>) -> tensor<64x32xf32>
                    loom.semaphore_give %84 : memref<64x32xf16>
                    loom.semaphore_give %72 : memref<64x64xf16>
                    scf.yield %111 : tensor<64x32xf32>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %35 : memref<64xf32>
                  %86 = loom.alloc [1] on @L1 : memref<f16>
                  %87 = loom.semaphore_take %86 : memref<f16> -> memref<f16>
                  %88 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%88], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                  %89 = arith.addi %29, %c3 : index
                  loom.copy %reinterpret_cast_6, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%29, %c0], LR : [%89, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %90 = loom.bufferize_to_tensor %87[] : memref<f16> -> tensor<f16>
                  %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
                  %93 = loom.init_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%94], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                  %95 = loom.bufferize_to_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %95, %90 : tensor<64x32xf32>, tensor<64x32xf16>, tensor<f16>) outs(%93 : tensor<64x32xf16>) {
                  ^bb0(%in: f32, %in_9: f16, %in_10: f16, %out: f16):
                    %99 = arith.extf %in_10 : f16 to f32
                    %100 = arith.extf %in_9 : f16 to f32
                    %101 = arith.mulf %100, %99 : f32
                    %102 = arith.addf %in, %101 : f32
                    %103 = arith.truncf %102 : f32 to f16
                    linalg.yield %103 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %87 : memref<f16>
                  loom.semaphore_give %63 : memref<64x32xf32>
                  %97 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%97], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  %98 = loom.bufferize_to_memref %96 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %98, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.semaphore_give %92 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c8 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg11, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg8, %c4 : index
                  %30 = arith.addi %28, %29 : index
                  %31 = arith.muli %arg12, %c4 : index
                  %32 = arith.addi %31, %c3 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%30, %31], LR : [%30, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %33 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %34 = loom.alloc [64] on @L1 : memref<64xf32>
                  %35 = loom.semaphore_take %34 : memref<64xf32> -> memref<64xf32>
                  %36 = loom.init_tensor %35[64] : memref<64xf32> -> tensor<64xf32>
                  %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%33 : tensor<64xf16>) outs(%36 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %99 = arith.extf %in : f16 to f32
                    linalg.yield %99 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %25 : memref<64xf16>
                  %38 = loom.alloc [64] on @L1 : memref<64xf32>
                  %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
                  %40 = loom.init_tensor %39[64] : memref<64xf32> -> tensor<64xf32>
                  %41 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%37 : tensor<64xf32>) outs(%40 : tensor<64xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %99 = arith.truncf %cst_0 : f64 to f32
                    %100 = arith.mulf %in, %99 : f32
                    %101 = math.powf %cst, %100 : f32
                    linalg.yield %101 : f32
                  } -> tensor<64xf32>
                  %42 = arith.muli %arg12, %c1024 : index
                  %43 = arith.divui %21, %c16 : index
                  %44 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
                  %45 = loom.semaphore_take %44 : memref<64x16xf16> -> memref<64x16xf16>
                  %c0_2 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 16 + d2 * 16 + d3)>(%arg11, %42, %43, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %47 = arith.addi %27, %29 : index
                  %48 = arith.addi %27, %c1 : index
                  %49 = arith.addi %48, %29 : index
                  loom.copy %reinterpret_cast_3, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
                  %50 = loom.bufferize_to_tensor %45[64, 16] : memref<64x16xf16> -> tensor<64x16xf16>
                  %51 = arith.muli %arg10, %c32 : index
                  %52 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
                  %53 = loom.semaphore_take %52 : memref<32x16xf16> -> memref<32x16xf16>
                  %c0_4 = arith.constant 0 : index
                  %54 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 131072 + d1 * 16384 + d2 * 1024 + d3 * 16 + d4)>(%arg11, %22, %21, %51, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %55 = arith.addi %arg10, %31 : index
                  loom.copy %reinterpret_cast_5, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
                  %56 = loom.bufferize_to_tensor %53[32, 16] : memref<32x16xf16> -> tensor<32x16xf16>
                  %57 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
                  %58 = loom.semaphore_take %57 : memref<16x32xf16> -> memref<16x32xf16>
                  %59 = loom.init_tensor %58[16, 32] : memref<16x32xf16> -> tensor<16x32xf16>
                  %transposed = linalg.transpose ins(%56 : tensor<32x16xf16>) outs(%59 : tensor<16x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %53 : memref<32x16xf16>
                  %60 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
                  %61 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %62 = loom.init_tensor %61[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %63 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %64 = loom.init_tensor %63[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %65 = linalg.fill ins(%cst_1 : f32) outs(%62 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  %66 = linalg.matmul ins(%50, %transposed : tensor<64x16xf16>, tensor<16x32xf16>) outs(%65 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  loom.semaphore_give %58 : memref<16x32xf16>
                  loom.semaphore_give %45 : memref<64x16xf16>
                  %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %41 : tensor<64x32xf32>, tensor<64xf32>) outs(%64 : tensor<64x32xf32>) {
                  ^bb0(%in: f32, %in_9: f32, %out: f32):
                    %99 = arith.mulf %in, %in_9 : f32
                    linalg.yield %99 : f32
                  } -> tensor<64x32xf32>
                  loom.semaphore_give %61 : memref<64x32xf32>
                  loom.semaphore_give %39 : memref<64xf32>
                  %68 = arith.addi %20, %c1 : index
                  %69 = arith.muli %68, %c64 : index
                  %70 = arith.ceildivui %69, %c64 : index
                  %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
                  %73 = loom.init_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %74 = loom.alloc [64] on @L1 : memref<64xf16>
                  %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %76 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %77 = loom.alloc [64] on @L1 : memref<64xf32>
                  %78 = loom.semaphore_take %77 : memref<64xf32> -> memref<64xf32>
                  %79 = loom.init_tensor %78[64] : memref<64xf32> -> tensor<64xf32>
                  %80 = loom.alloc [64] on @L1 : memref<64xf32>
                  %81 = loom.semaphore_take %80 : memref<64xf32> -> memref<64xf32>
                  %82 = loom.init_tensor %81[64] : memref<64xf32> -> tensor<64xf32>
                  %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
                  %85 = scf.for %arg14 = %c0 to %70 step %c1 iter_args(%arg15 = %67) -> (tensor<64x32xf32>) {
                    %99 = arith.muli %arg14, %c64 : index
                    %100 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %43, %23, %99)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%100], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%30, %31], LR : [%30, %32]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %101 = loom.bufferize_to_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %102 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %103 = loom.bufferize_to_tensor %76[64] : memref<64xf16> -> tensor<64xf16>
                    %104 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%103 : tensor<64xf16>) outs(%79 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %76 : memref<64xf16>
                    %105 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%105], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %106 = loom.bufferize_to_tensor %75[64] : memref<64xf16> -> tensor<64xf16>
                    %107 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%106 : tensor<64xf16>) outs(%82 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %75 : memref<64xf16>
                    %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %37, %104, %107 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%73 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f32, %in_14: f32, %in_15: f32, %out: f16):
                      %112 = arith.truncf %cst_0 : f64 to f32
                      %113 = arith.mulf %in_14, %112 : f32
                      %114 = arith.mulf %in_13, %112 : f32
                      %115 = arith.subf %114, %113 : f32
                      %116 = math.powf %cst, %115 : f32
                      %117 = arith.extf %in : f16 to f32
                      %118 = arith.mulf %117, %116 : f32
                      %119 = arith.mulf %118, %in_15 : f32
                      %120 = arith.truncf %119 : f32 to f16
                      linalg.yield %120 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %81 : memref<64xf32>
                    loom.semaphore_give %78 : memref<64xf32>
                    %109 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%109], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                    %110 = loom.bufferize_to_tensor %84[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %111 = linalg.matmul ins(%108, %110 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf32>) -> tensor<64x32xf32>
                    loom.semaphore_give %84 : memref<64x32xf16>
                    loom.semaphore_give %72 : memref<64x64xf16>
                    scf.yield %111 : tensor<64x32xf32>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %35 : memref<64xf32>
                  %86 = loom.alloc [1] on @L1 : memref<f16>
                  %87 = loom.semaphore_take %86 : memref<f16> -> memref<f16>
                  %88 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%88], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                  %89 = arith.addi %29, %c3 : index
                  loom.copy %reinterpret_cast_6, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%29, %c0], LR : [%89, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %90 = loom.bufferize_to_tensor %87[] : memref<f16> -> tensor<f16>
                  %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
                  %93 = loom.init_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%94], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                  %95 = loom.bufferize_to_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %95, %90 : tensor<64x32xf32>, tensor<64x32xf16>, tensor<f16>) outs(%93 : tensor<64x32xf16>) {
                  ^bb0(%in: f32, %in_9: f16, %in_10: f16, %out: f16):
                    %99 = arith.extf %in_10 : f16 to f32
                    %100 = arith.extf %in_9 : f16 to f32
                    %101 = arith.mulf %100, %99 : f32
                    %102 = arith.addf %in, %101 : f32
                    %103 = arith.truncf %102 : f32 to f16
                    linalg.yield %103 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %87 : memref<f16>
                  loom.semaphore_give %63 : memref<64x32xf32>
                  %97 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%97], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  %98 = loom.bufferize_to_memref %96 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %98, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.semaphore_give %92 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 16 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_y_level0_bc4_dim_x_level0_bc2_dim_x_level1_bc4_dim_y_level1_bc8_dim_x_level0_bc2_dim_x_level0_bc2__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x16x8x256xf16>, %arg2: memref<2x16x8x256xf16>, %arg3: memref<2x2048x16x64xf16>, %arg4: memref<2x2048x1x16xf16>, %arg5: memref<2x8x16x64x16xf16>, %arg6: memref<16xf16>, %arg7: memref<2x2048x16x64xf16>) {
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 2.000000e+00 : f32
      %cst_0 = arith.constant 1.44269504 : f64
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      affine.parallel (%arg8) = (0) to (2) {
        affine.parallel (%arg9) = (0) to (2) {
          affine.parallel (%arg10) = (0) to (4) {
            affine.parallel (%arg11) = (0) to (2) {
              affine.parallel (%arg12) = (0) to (2) {
                scf.for %arg13 = %c0 to %c2 step %c1 {
                  %20 = affine.apply affine_map<(d0, d1) -> (d0 + d1 * 2)>(%arg9, %arg13)
                  %21 = arith.muli %arg8, %c8 : index
                  %22 = arith.muli %arg12, %c4 : index
                  %23 = arith.muli %20, %c64 : index
                  %24 = loom.alloc [64] on @L1 : memref<64xf16>
                  %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
                  %26 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %23)
                  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%26], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                  %27 = arith.muli %arg12, %c2 : index
                  %28 = arith.addi %arg9, %27 : index
                  %29 = arith.muli %arg8, %c4 : index
                  %30 = arith.addi %28, %29 : index
                  %31 = arith.muli %arg11, %c4 : index
                  %32 = arith.addi %31, %c3 : index
                  loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%30, %31], LR : [%30, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                  %33 = loom.bufferize_to_tensor %25[64] : memref<64xf16> -> tensor<64xf16>
                  %34 = loom.alloc [64] on @L1 : memref<64xf32>
                  %35 = loom.semaphore_take %34 : memref<64xf32> -> memref<64xf32>
                  %36 = loom.init_tensor %35[64] : memref<64xf32> -> tensor<64xf32>
                  %37 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%33 : tensor<64xf16>) outs(%36 : tensor<64xf32>) {
                  ^bb0(%in: f16, %out: f32):
                    %99 = arith.extf %in : f16 to f32
                    linalg.yield %99 : f32
                  } -> tensor<64xf32>
                  loom.semaphore_give %25 : memref<64xf16>
                  %38 = loom.alloc [64] on @L1 : memref<64xf32>
                  %39 = loom.semaphore_take %38 : memref<64xf32> -> memref<64xf32>
                  %40 = loom.init_tensor %39[64] : memref<64xf32> -> tensor<64xf32>
                  %41 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%37 : tensor<64xf32>) outs(%40 : tensor<64xf32>) {
                  ^bb0(%in: f32, %out: f32):
                    %99 = arith.truncf %cst_0 : f64 to f32
                    %100 = arith.mulf %in, %99 : f32
                    %101 = math.powf %cst, %100 : f32
                    linalg.yield %101 : f32
                  } -> tensor<64xf32>
                  %42 = arith.muli %arg12, %c1024 : index
                  %43 = arith.divui %21, %c16 : index
                  %44 = loom.alloc [64, 16] on @L1 : memref<64x16xf16>
                  %45 = loom.semaphore_take %44 : memref<64x16xf16> -> memref<64x16xf16>
                  %c0_2 = arith.constant 0 : index
                  %46 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 16 + d2 * 16 + d3)>(%arg11, %42, %43, %c0_2)
                  %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%46], sizes: [64, 16], strides: [16, 1] : memref<2x2048x1x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %47 = arith.addi %27, %29 : index
                  %48 = arith.addi %27, %c1 : index
                  %49 = arith.addi %48, %29 : index
                  loom.copy %reinterpret_cast_3, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<64x16xf16>
                  %50 = loom.bufferize_to_tensor %45[64, 16] : memref<64x16xf16> -> tensor<64x16xf16>
                  %51 = arith.muli %arg10, %c32 : index
                  %52 = loom.alloc [32, 16] on @L1 : memref<32x16xf16>
                  %53 = loom.semaphore_take %52 : memref<32x16xf16> -> memref<32x16xf16>
                  %c0_4 = arith.constant 0 : index
                  %54 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 131072 + d1 * 16384 + d2 * 1024 + d3 * 16 + d4)>(%arg11, %22, %21, %51, %c0_4)
                  %reinterpret_cast_5 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [32, 16], strides: [16, 1] : memref<2x8x16x64x16xf16> to memref<?x16xf16, strided<[16, 1], offset: ?>>
                  %55 = arith.addi %arg10, %31 : index
                  loom.copy %reinterpret_cast_5, %53 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x16xf16, strided<[16, 1], offset: ?>> to memref<32x16xf16>
                  %56 = loom.bufferize_to_tensor %53[32, 16] : memref<32x16xf16> -> tensor<32x16xf16>
                  %57 = loom.alloc [16, 32] on @L1 : memref<16x32xf16>
                  %58 = loom.semaphore_take %57 : memref<16x32xf16> -> memref<16x32xf16>
                  %59 = loom.init_tensor %58[16, 32] : memref<16x32xf16> -> tensor<16x32xf16>
                  %transposed = linalg.transpose ins(%56 : tensor<32x16xf16>) outs(%59 : tensor<16x32xf16>) permutation = [1, 0] 
                  loom.semaphore_give %53 : memref<32x16xf16>
                  %60 = loom.alloc [64, 32] on @L1 : memref<64x32xf32>
                  %61 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %62 = loom.init_tensor %61[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %63 = loom.semaphore_take %60 : memref<64x32xf32> -> memref<64x32xf32>
                  %64 = loom.init_tensor %63[64, 32] : memref<64x32xf32> -> tensor<64x32xf32>
                  %65 = linalg.fill ins(%cst_1 : f32) outs(%62 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  %66 = linalg.matmul ins(%50, %transposed : tensor<64x16xf16>, tensor<16x32xf16>) outs(%65 : tensor<64x32xf32>) -> tensor<64x32xf32>
                  loom.semaphore_give %58 : memref<16x32xf16>
                  loom.semaphore_give %45 : memref<64x16xf16>
                  %67 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%66, %41 : tensor<64x32xf32>, tensor<64xf32>) outs(%64 : tensor<64x32xf32>) {
                  ^bb0(%in: f32, %in_9: f32, %out: f32):
                    %99 = arith.mulf %in, %in_9 : f32
                    linalg.yield %99 : f32
                  } -> tensor<64x32xf32>
                  loom.semaphore_give %61 : memref<64x32xf32>
                  loom.semaphore_give %39 : memref<64xf32>
                  %68 = arith.addi %20, %c1 : index
                  %69 = arith.muli %68, %c64 : index
                  %70 = arith.ceildivui %69, %c64 : index
                  %71 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
                  %72 = loom.semaphore_take %71 : memref<64x64xf16> -> memref<64x64xf16>
                  %73 = loom.init_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                  %74 = loom.alloc [64] on @L1 : memref<64xf16>
                  %75 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %76 = loom.semaphore_take %74 : memref<64xf16> -> memref<64xf16>
                  %77 = loom.alloc [64] on @L1 : memref<64xf32>
                  %78 = loom.semaphore_take %77 : memref<64xf32> -> memref<64xf32>
                  %79 = loom.init_tensor %78[64] : memref<64xf32> -> tensor<64xf32>
                  %80 = loom.alloc [64] on @L1 : memref<64xf32>
                  %81 = loom.semaphore_take %80 : memref<64xf32> -> memref<64xf32>
                  %82 = loom.init_tensor %81[64] : memref<64xf32> -> tensor<64xf32>
                  %83 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %84 = loom.semaphore_take %83 : memref<64x32xf16> -> memref<64x32xf16>
                  %85 = scf.for %arg14 = %c0 to %70 step %c1 iter_args(%arg15 = %67) -> (tensor<64x32xf32>) {
                    %99 = arith.muli %arg14, %c64 : index
                    %100 = affine.apply affine_map<(d0, d1, d2, d3, d4) -> (d0 * 524288 + d1 * 65536 + d2 * 65536 + d3 * 256 + d4)>(%arg11, %22, %43, %23, %99)
                    %reinterpret_cast_9 = memref.reinterpret_cast %arg0 to offset: [%100], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
                    loom.copy %reinterpret_cast_9, %72 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%30, %31], LR : [%30, %32]) : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
                    %101 = loom.bufferize_to_tensor %72[64, 64] : memref<64x64xf16> -> tensor<64x64xf16>
                    %102 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_10 = memref.reinterpret_cast %arg1 to offset: [%102], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_10, %76 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %103 = loom.bufferize_to_tensor %76[64] : memref<64xf16> -> tensor<64xf16>
                    %104 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%103 : tensor<64xf16>) outs(%79 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %76 : memref<64xf16>
                    %105 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 2048 + d2 * 256 + d3)>(%arg11, %21, %22, %99)
                    %reinterpret_cast_11 = memref.reinterpret_cast %arg2 to offset: [%105], sizes: [64], strides: [1] : memref<2x16x8x256xf16> to memref<?xf16, strided<[1], offset: ?>>
                    loom.copy %reinterpret_cast_11, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 4] region : (UL : [%47, %31], LR : [%49, %32]) : memref<?xf16, strided<[1], offset: ?>> to memref<64xf16>
                    %106 = loom.bufferize_to_tensor %75[64] : memref<64xf16> -> tensor<64xf16>
                    %107 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%106 : tensor<64xf16>) outs(%82 : tensor<64xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %112 = arith.extf %in : f16 to f32
                      linalg.yield %112 : f32
                    } -> tensor<64xf32>
                    loom.semaphore_give %75 : memref<64xf16>
                    %108 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%101, %37, %104, %107 : tensor<64x64xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%73 : tensor<64x64xf16>) {
                    ^bb0(%in: f16, %in_13: f32, %in_14: f32, %in_15: f32, %out: f16):
                      %112 = arith.truncf %cst_0 : f64 to f32
                      %113 = arith.mulf %in_14, %112 : f32
                      %114 = arith.mulf %in_13, %112 : f32
                      %115 = arith.subf %114, %113 : f32
                      %116 = math.powf %cst, %115 : f32
                      %117 = arith.extf %in : f16 to f32
                      %118 = arith.mulf %117, %116 : f32
                      %119 = arith.mulf %118, %in_15 : f32
                      %120 = arith.truncf %119 : f32 to f16
                      linalg.yield %120 : f16
                    } -> tensor<64x64xf16>
                    loom.semaphore_give %81 : memref<64xf32>
                    loom.semaphore_give %78 : memref<64xf32>
                    %109 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                    %reinterpret_cast_12 = memref.reinterpret_cast %arg3 to offset: [%109], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                    loom.copy %reinterpret_cast_12, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                    %110 = loom.bufferize_to_tensor %84[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                    %111 = linalg.matmul ins(%108, %110 : tensor<64x64xf16>, tensor<64x32xf16>) outs(%arg15 : tensor<64x32xf32>) -> tensor<64x32xf32>
                    loom.semaphore_give %84 : memref<64x32xf16>
                    loom.semaphore_give %72 : memref<64x64xf16>
                    scf.yield %111 : tensor<64x32xf32>
                  } {loom.iter_type = #loom.iter_type<sequential>}
                  loom.semaphore_give %35 : memref<64xf32>
                  %86 = loom.alloc [1] on @L1 : memref<f16>
                  %87 = loom.semaphore_take %86 : memref<f16> -> memref<f16>
                  %88 = affine.apply affine_map<(d0) -> (d0)>(%21)
                  %reinterpret_cast_6 = memref.reinterpret_cast %arg6 to offset: [%88], sizes: [], strides: [] : memref<16xf16> to memref<f16, strided<[], offset: ?>>
                  %89 = arith.addi %29, %c3 : index
                  loom.copy %reinterpret_cast_6, %87 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%29, %c0], LR : [%89, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
                  %90 = loom.bufferize_to_tensor %87[] : memref<f16> -> tensor<f16>
                  %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
                  %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
                  %93 = loom.init_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %94 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                  %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%94], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.copy %reinterpret_cast_7, %92 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<?x?xf16, strided<[1024, 1], offset: ?>> to memref<64x32xf16>
                  %95 = loom.bufferize_to_tensor %92[64, 32] : memref<64x32xf16> -> tensor<64x32xf16>
                  %96 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%85, %95, %90 : tensor<64x32xf32>, tensor<64x32xf16>, tensor<f16>) outs(%93 : tensor<64x32xf16>) {
                  ^bb0(%in: f32, %in_9: f16, %in_10: f16, %out: f16):
                    %99 = arith.extf %in_10 : f16 to f32
                    %100 = arith.extf %in_9 : f16 to f32
                    %101 = arith.mulf %100, %99 : f32
                    %102 = arith.addf %in, %101 : f32
                    %103 = arith.truncf %102 : f32 to f16
                    linalg.yield %103 : f16
                  } -> tensor<64x32xf16>
                  loom.semaphore_give %87 : memref<f16>
                  loom.semaphore_give %63 : memref<64x32xf32>
                  %97 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 * 2097152 + d1 * 1024 + d2 * 64 + d3)>(%arg11, %42, %21, %51)
                  %reinterpret_cast_8 = memref.reinterpret_cast %arg7 to offset: [%97], sizes: [64, 32], strides: [1024, 1] : memref<2x2048x16x64xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  %98 = loom.bufferize_to_memref %96 : tensor<64x32xf16> -> memref<64x32xf16>
                  loom.copy %98, %reinterpret_cast_8 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [2, 1] region : (UL : [%47, %55], LR : [%49, %55]) : memref<64x32xf16> to memref<?x?xf16, strided<[1024, 1], offset: ?>>
                  loom.semaphore_give %92 : memref<64x32xf16>
                } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
              } {loom.block_sym = @tile_c, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_x}
            } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 1 : i64, loom.physical_dim = @dim_y}
          } {loom.block_sym = @tile_n, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_y}
        } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 0 : i64, loom.physical_dim = @dim_x}
      } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<spatial>, loom.logical_level = 2 : i64, loom.physical_dim = @dim_x}
      return
    }
  }
}
