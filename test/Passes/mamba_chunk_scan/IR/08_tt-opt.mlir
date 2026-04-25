module attributes {loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c4, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %29 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %30 = arith.muli %21, %c16384 : index
            %31 = arith.addi %29, %30 : index
            %32 = arith.muli %arg12, %c1024 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %35 = arith.muli %arg11, %c2 : index
            %36 = arith.addi %arg9, %35 : index
            %37 = arith.muli %arg12, %c2 : index
            %38 = arith.muli %arg8, %c4 : index
            %39 = arith.addi %37, %38 : index
            %40 = arith.addi %37, %c1 : index
            %41 = arith.addi %40, %38 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%36, %39], LR : [%36, %41]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%27 : memref<64xf16>) outs(%28 : memref<64xf16>)
            loom.semaphore_give %27 : memref<64xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.broadcast ins(%28 : memref<64xf16>) outs(%44 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %46 = arith.addi %25, %32 : index
            %47 = arith.divui %24, %c64 : index
            %48 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %49 = loom.semaphore_take %48 : memref<64x64xf16> -> memref<64x64xf16>
            %50 = loom.semaphore_take %48 : memref<64x64xf16> -> memref<64x64xf16>
            %51 = arith.muli %46, %c64 overflow<nsw> : index
            %52 = arith.addi %29, %51 : index
            %53 = arith.muli %47, %c64 overflow<nsw> : index
            %54 = arith.addi %52, %53 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%36, %39], LR : [%36, %41]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            loom.sync ins(%50 : memref<64x64xf16>) outs(%49 : memref<64x64xf16>)
            loom.semaphore_give %50 : memref<64x64xf16>
            %55 = arith.muli %arg10, %c32 : index
            %56 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %57 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
            %58 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
            %59 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %60 = arith.muli %arg12, %c1048576 : index
            %61 = arith.addi %59, %60 : index
            %62 = arith.muli %21, %c32768 : index
            %63 = arith.addi %61, %62 : index
            %64 = arith.addi %63, %55 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%64], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %65 = arith.addi %35, %c1 : index
            %66 = arith.addi %arg10, %37 : index
            %67 = arith.addi %66, %38 : index
            loom.copy %reinterpret_cast_1, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%35, %67], LR : [%65, %67]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%58 : memref<64x32xf16>) outs(%57 : memref<64x32xf16>)
            loom.semaphore_give %58 : memref<64x32xf16>
            %68 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %69 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            %70 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%49, %57 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %57 : memref<64x32xf16>
            loom.semaphore_give %49 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %45 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%70 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %104 = math.exp %in_5 : f16
              %105 = arith.mulf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
            loom.semaphore_give %44 : memref<64x32xf16>
            %71 = arith.addi %23, %c1 : index
            %72 = arith.muli %71, %c64 : index
            %73 = arith.ceildivui %72, %c64 : index
            %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %76 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %77 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %78 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %79 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %80 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %81 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %82 = loom.alloc [64] on @L1 : memref<64xf16>
            %83 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %84 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %85 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %86 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %87 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %88 = loom.semaphore_take %87 : memref<32x64xf16> -> memref<32x64xf16>
            %89 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %90 = loom.semaphore_take %89 : memref<32x64xf16> -> memref<32x64xf16>
            %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            %93 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %73 step %c1 {
              %104 = arith.muli %arg15, %c64 : index
              %105 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %106 = arith.muli %arg12, %c262144 : index
              %107 = arith.addi %105, %106 : index
              %108 = arith.muli %47, %c65536 overflow<nsw> : index
              %109 = arith.addi %107, %108 : index
              %110 = arith.muli %23, %c16384 : index
              %111 = arith.addi %109, %110 : index
              %112 = arith.addi %111, %104 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%112], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              loom.sync ins(%81 : memref<64x64xf16>) outs(%80 : memref<64x64xf16>)
              loom.semaphore_give %81 : memref<64x64xf16>
              %113 = arith.addi %33, %104 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%86 : memref<64xf16>) outs(%85 : memref<64xf16>)
              loom.semaphore_give %86 : memref<64xf16>
              %114 = loom.broadcast ins(%28 : memref<64xf16>) outs(%43 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %115 = loom.broadcast ins(%85 : memref<64xf16>) outs(%88 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %85 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%84 : memref<64xf16>) outs(%83 : memref<64xf16>)
              loom.semaphore_give %84 : memref<64xf16>
              %116 = loom.broadcast ins(%83 : memref<64xf16>) outs(%90 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %83 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80, %114, %115, %116 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%80 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %124 = arith.subf %in_9, %in_10 : f16
                %125 = math.exp %124 : f16
                %126 = arith.mulf %in, %125 : f16
                %127 = arith.mulf %126, %in_11 : f16
                linalg.yield %127 : f16
              }
              loom.semaphore_give %90 : memref<32x64xf16>
              loom.semaphore_give %88 : memref<32x64xf16>
              loom.semaphore_give %43 : memref<64x32xf16>
              %117 = arith.addi %104, %32 : index
              %118 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %119 = arith.muli %117, %c4096 overflow<nsw> : index
              %120 = arith.addi %118, %119 : index
              %121 = arith.muli %21, %c512 : index
              %122 = arith.addi %120, %121 : index
              %123 = arith.addi %122, %55 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%123], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              loom.sync ins(%93 : memref<64x32xf16>) outs(%92 : memref<64x32xf16>)
              loom.semaphore_give %93 : memref<64x32xf16>
              loom.matmul ins(%80, %92 : memref<64x64xf16>, memref<64x32xf16>) outs(%78 : memref<64x32xf16>)
              loom.semaphore_give %92 : memref<64x32xf16>
              loom.semaphore_give %80 : memref<64x64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %78 : memref<64x32xf16>, memref<64x32xf16>) outs(%70 : memref<64x32xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.addf %in, %in_9 : f16
                linalg.yield %124 : f16
              }
              loom.semaphore_give %78 : memref<64x32xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %28 : memref<64xf16>
            %94 = loom.alloc [1] on @L1 : memref<f16>
            %95 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %96 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %97 = arith.addi %38, %c3 : index
            loom.copy %reinterpret_cast_2, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %38], LR : [%c7, %97]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            loom.sync ins(%96 : memref<f16>) outs(%95 : memref<f16>)
            loom.semaphore_give %96 : memref<f16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %46, %c4096 overflow<nsw> : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %21, %c512 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %55 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%77 : memref<64x32xf16>) outs(%76 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %76, %95 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%76 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %104 = arith.mulf %in_5, %in_6 : f16
              %105 = arith.addf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %95 : memref<f16>
            loom.semaphore_give %70 : memref<64x32xf16>
            loom.sync ins(%76 : memref<64x32xf16>) outs(%75 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %75, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %75 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x4_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c2, %c4) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %29 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %30 = arith.muli %21, %c16384 : index
            %31 = arith.addi %29, %30 : index
            %32 = arith.muli %arg12, %c1024 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %35 = arith.muli %arg12, %c2 : index
            %36 = arith.addi %arg9, %35 : index
            %37 = arith.muli %arg11, %c2 : index
            %38 = arith.muli %arg8, %c4 : index
            %39 = arith.addi %37, %38 : index
            %40 = arith.addi %37, %c1 : index
            %41 = arith.addi %40, %38 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%36, %39], LR : [%36, %41]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%27 : memref<64xf16>) outs(%28 : memref<64xf16>)
            loom.semaphore_give %27 : memref<64xf16>
            %42 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %43 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.semaphore_take %42 : memref<64x32xf16> -> memref<64x32xf16>
            %45 = loom.broadcast ins(%28 : memref<64xf16>) outs(%44 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %46 = arith.addi %25, %32 : index
            %47 = arith.divui %24, %c64 : index
            %48 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %49 = loom.semaphore_take %48 : memref<64x64xf16> -> memref<64x64xf16>
            %50 = loom.semaphore_take %48 : memref<64x64xf16> -> memref<64x64xf16>
            %51 = arith.muli %46, %c64 overflow<nsw> : index
            %52 = arith.addi %29, %51 : index
            %53 = arith.muli %47, %c64 overflow<nsw> : index
            %54 = arith.addi %52, %53 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%54], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%36, %39], LR : [%36, %41]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            loom.sync ins(%50 : memref<64x64xf16>) outs(%49 : memref<64x64xf16>)
            loom.semaphore_give %50 : memref<64x64xf16>
            %55 = arith.muli %arg10, %c32 : index
            %56 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %57 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
            %58 = loom.semaphore_take %56 : memref<64x32xf16> -> memref<64x32xf16>
            %59 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %60 = arith.muli %arg12, %c1048576 : index
            %61 = arith.addi %59, %60 : index
            %62 = arith.muli %21, %c32768 : index
            %63 = arith.addi %61, %62 : index
            %64 = arith.addi %63, %55 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%64], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %65 = arith.addi %35, %c1 : index
            %66 = arith.addi %arg10, %37 : index
            %67 = arith.addi %66, %38 : index
            loom.copy %reinterpret_cast_1, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%35, %67], LR : [%65, %67]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%58 : memref<64x32xf16>) outs(%57 : memref<64x32xf16>)
            loom.semaphore_give %58 : memref<64x32xf16>
            %68 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %69 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            %70 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%49, %57 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %57 : memref<64x32xf16>
            loom.semaphore_give %49 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %45 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%70 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %104 = math.exp %in_5 : f16
              %105 = arith.mulf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
            loom.semaphore_give %44 : memref<64x32xf16>
            %71 = arith.addi %23, %c1 : index
            %72 = arith.muli %71, %c64 : index
            %73 = arith.ceildivui %72, %c64 : index
            %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %76 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %77 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %78 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %79 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %80 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %81 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %82 = loom.alloc [64] on @L1 : memref<64xf16>
            %83 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %84 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %85 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %86 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %87 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %88 = loom.semaphore_take %87 : memref<32x64xf16> -> memref<32x64xf16>
            %89 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %90 = loom.semaphore_take %89 : memref<32x64xf16> -> memref<32x64xf16>
            %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            %93 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %73 step %c1 {
              %104 = arith.muli %arg15, %c64 : index
              %105 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %106 = arith.muli %arg12, %c262144 : index
              %107 = arith.addi %105, %106 : index
              %108 = arith.muli %47, %c65536 overflow<nsw> : index
              %109 = arith.addi %107, %108 : index
              %110 = arith.muli %23, %c16384 : index
              %111 = arith.addi %109, %110 : index
              %112 = arith.addi %111, %104 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%112], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              loom.sync ins(%81 : memref<64x64xf16>) outs(%80 : memref<64x64xf16>)
              loom.semaphore_give %81 : memref<64x64xf16>
              %113 = arith.addi %33, %104 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%86 : memref<64xf16>) outs(%85 : memref<64xf16>)
              loom.semaphore_give %86 : memref<64xf16>
              %114 = loom.broadcast ins(%28 : memref<64xf16>) outs(%43 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %115 = loom.broadcast ins(%85 : memref<64xf16>) outs(%88 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %85 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%84 : memref<64xf16>) outs(%83 : memref<64xf16>)
              loom.semaphore_give %84 : memref<64xf16>
              %116 = loom.broadcast ins(%83 : memref<64xf16>) outs(%90 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %83 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80, %114, %115, %116 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%80 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %124 = arith.subf %in_9, %in_10 : f16
                %125 = math.exp %124 : f16
                %126 = arith.mulf %in, %125 : f16
                %127 = arith.mulf %126, %in_11 : f16
                linalg.yield %127 : f16
              }
              loom.semaphore_give %90 : memref<32x64xf16>
              loom.semaphore_give %88 : memref<32x64xf16>
              loom.semaphore_give %43 : memref<64x32xf16>
              %117 = arith.addi %104, %32 : index
              %118 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %119 = arith.muli %117, %c4096 overflow<nsw> : index
              %120 = arith.addi %118, %119 : index
              %121 = arith.muli %21, %c512 : index
              %122 = arith.addi %120, %121 : index
              %123 = arith.addi %122, %55 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%123], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              loom.sync ins(%93 : memref<64x32xf16>) outs(%92 : memref<64x32xf16>)
              loom.semaphore_give %93 : memref<64x32xf16>
              loom.matmul ins(%80, %92 : memref<64x64xf16>, memref<64x32xf16>) outs(%78 : memref<64x32xf16>)
              loom.semaphore_give %92 : memref<64x32xf16>
              loom.semaphore_give %80 : memref<64x64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %78 : memref<64x32xf16>, memref<64x32xf16>) outs(%70 : memref<64x32xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.addf %in, %in_9 : f16
                linalg.yield %124 : f16
              }
              loom.semaphore_give %78 : memref<64x32xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %28 : memref<64xf16>
            %94 = loom.alloc [1] on @L1 : memref<f16>
            %95 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %96 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %97 = arith.addi %38, %c3 : index
            loom.copy %reinterpret_cast_2, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %38], LR : [%c7, %97]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            loom.sync ins(%96 : memref<f16>) outs(%95 : memref<f16>)
            loom.semaphore_give %96 : memref<f16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %46, %c4096 overflow<nsw> : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %21, %c512 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %55 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%77 : memref<64x32xf16>) outs(%76 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %76, %95 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%76 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %104 = arith.mulf %in_5, %in_6 : f16
              %105 = arith.addf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %95 : memref<f16>
            loom.semaphore_give %70 : memref<64x32xf16>
            loom.sync ins(%76 : memref<64x32xf16>) outs(%75 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %75, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%36, %67], LR : [%36, %67]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %75 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c4, %c2, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg8, %20 : index
          %22 = arith.muli %21, %c8 : index
          %23 = arith.muli %arg9, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %27 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %28 = arith.muli %21, %c16384 : index
          %29 = arith.addi %27, %28 : index
          %30 = arith.muli %arg12, %c1024 : index
          %31 = arith.addi %29, %30 : index
          %32 = arith.addi %31, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %33 = arith.muli %arg11, %c4 : index
          %34 = arith.addi %arg9, %33 : index
          %35 = arith.muli %arg12, %c2 : index
          %36 = arith.muli %arg8, %c4 : index
          %37 = arith.addi %35, %36 : index
          %38 = arith.addi %35, %c1 : index
          %39 = arith.addi %38, %36 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%34, %37], LR : [%34, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          loom.sync ins(%25 : memref<64xf16>) outs(%26 : memref<64xf16>)
          loom.semaphore_give %25 : memref<64xf16>
          %40 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %41 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
          %42 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
          %43 = loom.broadcast ins(%26 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
          %44 = arith.addi %23, %30 : index
          %45 = arith.divui %22, %c64 : index
          %46 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %47 = loom.semaphore_take %46 : memref<64x64xf16> -> memref<64x64xf16>
          %48 = loom.semaphore_take %46 : memref<64x64xf16> -> memref<64x64xf16>
          %49 = arith.muli %44, %c64 overflow<nsw> : index
          %50 = arith.addi %27, %49 : index
          %51 = arith.muli %45, %c64 overflow<nsw> : index
          %52 = arith.addi %50, %51 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%34, %37], LR : [%34, %39]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          loom.sync ins(%48 : memref<64x64xf16>) outs(%47 : memref<64x64xf16>)
          loom.semaphore_give %48 : memref<64x64xf16>
          %53 = arith.muli %arg10, %c32 : index
          %54 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %55 = loom.semaphore_take %54 : memref<64x32xf16> -> memref<64x32xf16>
          %56 = loom.semaphore_take %54 : memref<64x32xf16> -> memref<64x32xf16>
          %57 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %58 = arith.muli %arg12, %c1048576 : index
          %59 = arith.addi %57, %58 : index
          %60 = arith.muli %21, %c32768 : index
          %61 = arith.addi %59, %60 : index
          %62 = arith.addi %61, %53 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%62], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
          %63 = arith.addi %33, %c3 : index
          %64 = arith.addi %arg10, %35 : index
          %65 = arith.addi %64, %36 : index
          loom.copy %reinterpret_cast_1, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%33, %65], LR : [%63, %65]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
          loom.sync ins(%56 : memref<64x32xf16>) outs(%55 : memref<64x32xf16>)
          loom.semaphore_give %56 : memref<64x32xf16>
          %66 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %67 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
          %68 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%47, %55 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
          loom.semaphore_give %55 : memref<64x32xf16>
          loom.semaphore_give %47 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %43 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%68 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %102 = math.exp %in_5 : f16
            %103 = arith.mulf %in, %102 : f16
            linalg.yield %103 : f16
          }
          loom.semaphore_give %67 : memref<64x32xf16>
          loom.semaphore_give %42 : memref<64x32xf16>
          %69 = arith.addi %arg9, %c1 : index
          %70 = arith.muli %69, %c64 : index
          %71 = arith.ceildivui %70, %c64 : index
          %72 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %73 = loom.semaphore_take %72 : memref<64x32xf16> -> memref<64x32xf16>
          %74 = loom.semaphore_take %72 : memref<64x32xf16> -> memref<64x32xf16>
          %75 = loom.semaphore_take %72 : memref<64x32xf16> -> memref<64x32xf16>
          %76 = loom.semaphore_take %72 : memref<64x32xf16> -> memref<64x32xf16>
          %77 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %78 = loom.semaphore_take %77 : memref<64x64xf16> -> memref<64x64xf16>
          %79 = loom.semaphore_take %77 : memref<64x64xf16> -> memref<64x64xf16>
          %80 = loom.alloc [64] on @L1 : memref<64xf16>
          %81 = loom.semaphore_take %80 : memref<64xf16> -> memref<64xf16>
          %82 = loom.semaphore_take %80 : memref<64xf16> -> memref<64xf16>
          %83 = loom.semaphore_take %80 : memref<64xf16> -> memref<64xf16>
          %84 = loom.semaphore_take %80 : memref<64xf16> -> memref<64xf16>
          %85 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %86 = loom.semaphore_take %85 : memref<32x64xf16> -> memref<32x64xf16>
          %87 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %88 = loom.semaphore_take %87 : memref<32x64xf16> -> memref<32x64xf16>
          %89 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %90 = loom.semaphore_take %89 : memref<64x32xf16> -> memref<64x32xf16>
          %91 = loom.semaphore_take %89 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %71 step %c1 {
            %102 = arith.muli %arg14, %c64 : index
            %103 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %104 = arith.muli %arg12, %c262144 : index
            %105 = arith.addi %103, %104 : index
            %106 = arith.muli %45, %c65536 overflow<nsw> : index
            %107 = arith.addi %105, %106 : index
            %108 = arith.muli %arg9, %c16384 : index
            %109 = arith.addi %107, %108 : index
            %110 = arith.addi %109, %102 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%110], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            loom.sync ins(%79 : memref<64x64xf16>) outs(%78 : memref<64x64xf16>)
            loom.semaphore_give %79 : memref<64x64xf16>
            %111 = arith.addi %31, %102 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%111], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_6, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%84 : memref<64xf16>) outs(%83 : memref<64xf16>)
            loom.semaphore_give %84 : memref<64xf16>
            %112 = loom.broadcast ins(%26 : memref<64xf16>) outs(%41 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            %113 = loom.broadcast ins(%83 : memref<64xf16>) outs(%86 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %83 : memref<64xf16>
            %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%111], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%82 : memref<64xf16>) outs(%81 : memref<64xf16>)
            loom.semaphore_give %82 : memref<64xf16>
            %114 = loom.broadcast ins(%81 : memref<64xf16>) outs(%88 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %81 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%78, %112, %113, %114 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%78 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
              %122 = arith.subf %in_9, %in_10 : f16
              %123 = math.exp %122 : f16
              %124 = arith.mulf %in, %123 : f16
              %125 = arith.mulf %124, %in_11 : f16
              linalg.yield %125 : f16
            }
            loom.semaphore_give %88 : memref<32x64xf16>
            loom.semaphore_give %86 : memref<32x64xf16>
            loom.semaphore_give %41 : memref<64x32xf16>
            %115 = arith.addi %102, %30 : index
            %116 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %117 = arith.muli %115, %c4096 overflow<nsw> : index
            %118 = arith.addi %116, %117 : index
            %119 = arith.muli %21, %c512 : index
            %120 = arith.addi %118, %119 : index
            %121 = arith.addi %120, %53 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%121], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_8, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%91 : memref<64x32xf16>) outs(%90 : memref<64x32xf16>)
            loom.semaphore_give %91 : memref<64x32xf16>
            loom.matmul ins(%78, %90 : memref<64x64xf16>, memref<64x32xf16>) outs(%76 : memref<64x32xf16>)
            loom.semaphore_give %90 : memref<64x32xf16>
            loom.semaphore_give %78 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%68, %76 : memref<64x32xf16>, memref<64x32xf16>) outs(%68 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %122 = arith.addf %in, %in_9 : f16
              linalg.yield %122 : f16
            }
            loom.semaphore_give %76 : memref<64x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %26 : memref<64xf16>
          %92 = loom.alloc [1] on @L1 : memref<f16>
          %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
          %94 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
          %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %95 = arith.addi %36, %c3 : index
          loom.copy %reinterpret_cast_2, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %36], LR : [%c7, %95]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          loom.sync ins(%94 : memref<f16>) outs(%93 : memref<f16>)
          loom.semaphore_give %94 : memref<f16>
          %96 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %97 = arith.muli %44, %c4096 overflow<nsw> : index
          %98 = arith.addi %96, %97 : index
          %99 = arith.muli %21, %c512 : index
          %100 = arith.addi %98, %99 : index
          %101 = arith.addi %100, %53 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%101], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          loom.sync ins(%75 : memref<64x32xf16>) outs(%74 : memref<64x32xf16>)
          loom.semaphore_give %75 : memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%68, %74, %93 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%74 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %102 = arith.mulf %in_5, %in_6 : f16
            %103 = arith.addf %in, %102 : f16
            linalg.yield %103 : f16
          }
          loom.semaphore_give %93 : memref<f16>
          loom.semaphore_give %68 : memref<64x32xf16>
          loom.sync ins(%74 : memref<64x32xf16>) outs(%73 : memref<64x32xf16>)
          loom.semaphore_give %74 : memref<64x32xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%101], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %73, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %73 : memref<64x32xf16>
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x4x2_y2y2y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc4_n_n_n_n_dim_x_level1_bc8_dim_y_level1_bc4_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c4, %c2, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          %20 = arith.muli %arg13, %c2 overflow<nsw> : index
          %21 = arith.addi %arg8, %20 : index
          %22 = arith.muli %21, %c8 : index
          %23 = arith.muli %arg9, %c64 : index
          %24 = loom.alloc [64] on @L1 : memref<64xf16>
          %25 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %26 = loom.semaphore_take %24 : memref<64xf16> -> memref<64xf16>
          %27 = arith.muli %arg11, %c131072 overflow<nsw> : index
          %28 = arith.muli %21, %c16384 : index
          %29 = arith.addi %27, %28 : index
          %30 = arith.muli %arg12, %c1024 : index
          %31 = arith.addi %29, %30 : index
          %32 = arith.addi %31, %23 : index
          %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%32], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
          %33 = arith.muli %arg12, %c4 : index
          %34 = arith.addi %arg9, %33 : index
          %35 = arith.muli %arg11, %c2 : index
          %36 = arith.muli %arg8, %c4 : index
          %37 = arith.addi %35, %36 : index
          %38 = arith.addi %35, %c1 : index
          %39 = arith.addi %38, %36 : index
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%34, %37], LR : [%34, %39]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
          loom.sync ins(%25 : memref<64xf16>) outs(%26 : memref<64xf16>)
          loom.semaphore_give %25 : memref<64xf16>
          %40 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %41 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
          %42 = loom.semaphore_take %40 : memref<64x32xf16> -> memref<64x32xf16>
          %43 = loom.broadcast ins(%26 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
          %44 = arith.addi %23, %30 : index
          %45 = arith.divui %22, %c64 : index
          %46 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %47 = loom.semaphore_take %46 : memref<64x64xf16> -> memref<64x64xf16>
          %48 = loom.semaphore_take %46 : memref<64x64xf16> -> memref<64x64xf16>
          %49 = arith.muli %44, %c64 overflow<nsw> : index
          %50 = arith.addi %27, %49 : index
          %51 = arith.muli %45, %c64 overflow<nsw> : index
          %52 = arith.addi %50, %51 : index
          %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%52], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          loom.copy %reinterpret_cast_0, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%34, %37], LR : [%34, %39]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
          loom.sync ins(%48 : memref<64x64xf16>) outs(%47 : memref<64x64xf16>)
          loom.semaphore_give %48 : memref<64x64xf16>
          %53 = arith.muli %arg10, %c32 : index
          %54 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %55 = loom.semaphore_take %54 : memref<64x32xf16> -> memref<64x32xf16>
          %56 = loom.semaphore_take %54 : memref<64x32xf16> -> memref<64x32xf16>
          %57 = arith.muli %arg11, %c2097152 overflow<nsw> : index
          %58 = arith.muli %arg12, %c1048576 : index
          %59 = arith.addi %57, %58 : index
          %60 = arith.muli %21, %c32768 : index
          %61 = arith.addi %59, %60 : index
          %62 = arith.addi %61, %53 : index
          %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%62], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
          %63 = arith.addi %33, %c3 : index
          %64 = arith.addi %arg10, %35 : index
          %65 = arith.addi %64, %36 : index
          loom.copy %reinterpret_cast_1, %56 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 1] region : (UL : [%33, %65], LR : [%63, %65]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
          loom.sync ins(%56 : memref<64x32xf16>) outs(%55 : memref<64x32xf16>)
          loom.semaphore_give %56 : memref<64x32xf16>
          %66 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %67 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
          %68 = loom.semaphore_take %66 : memref<64x32xf16> -> memref<64x32xf16>
          loom.matmul ins(%47, %55 : memref<64x64xf16>, memref<64x32xf16>) outs(%67 : memref<64x32xf16>)
          loom.semaphore_give %55 : memref<64x32xf16>
          loom.semaphore_give %47 : memref<64x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%67, %43 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%68 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %102 = math.exp %in_5 : f16
            %103 = arith.mulf %in, %102 : f16
            linalg.yield %103 : f16
          }
          loom.semaphore_give %67 : memref<64x32xf16>
          loom.semaphore_give %42 : memref<64x32xf16>
          %69 = arith.addi %arg9, %c1 : index
          %70 = arith.muli %69, %c64 : index
          %71 = arith.ceildivui %70, %c64 : index
          %72 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %73 = loom.semaphore_take %72 : memref<64x32xf16> -> memref<64x32xf16>
          %74 = loom.semaphore_take %72 : memref<64x32xf16> -> memref<64x32xf16>
          %75 = loom.semaphore_take %72 : memref<64x32xf16> -> memref<64x32xf16>
          %76 = loom.semaphore_take %72 : memref<64x32xf16> -> memref<64x32xf16>
          %77 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
          %78 = loom.semaphore_take %77 : memref<64x64xf16> -> memref<64x64xf16>
          %79 = loom.semaphore_take %77 : memref<64x64xf16> -> memref<64x64xf16>
          %80 = loom.alloc [64] on @L1 : memref<64xf16>
          %81 = loom.semaphore_take %80 : memref<64xf16> -> memref<64xf16>
          %82 = loom.semaphore_take %80 : memref<64xf16> -> memref<64xf16>
          %83 = loom.semaphore_take %80 : memref<64xf16> -> memref<64xf16>
          %84 = loom.semaphore_take %80 : memref<64xf16> -> memref<64xf16>
          %85 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %86 = loom.semaphore_take %85 : memref<32x64xf16> -> memref<32x64xf16>
          %87 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
          %88 = loom.semaphore_take %87 : memref<32x64xf16> -> memref<32x64xf16>
          %89 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
          %90 = loom.semaphore_take %89 : memref<64x32xf16> -> memref<64x32xf16>
          %91 = loom.semaphore_take %89 : memref<64x32xf16> -> memref<64x32xf16>
          scf.for %arg14 = %c0 to %71 step %c1 {
            %102 = arith.muli %arg14, %c64 : index
            %103 = arith.muli %arg11, %c524288 overflow<nsw> : index
            %104 = arith.muli %arg12, %c262144 : index
            %105 = arith.addi %103, %104 : index
            %106 = arith.muli %45, %c65536 overflow<nsw> : index
            %107 = arith.addi %105, %106 : index
            %108 = arith.muli %arg9, %c16384 : index
            %109 = arith.addi %107, %108 : index
            %110 = arith.addi %109, %102 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%110], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %79 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
            loom.sync ins(%79 : memref<64x64xf16>) outs(%78 : memref<64x64xf16>)
            loom.semaphore_give %79 : memref<64x64xf16>
            %111 = arith.addi %31, %102 : index
            %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%111], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_6, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%84 : memref<64xf16>) outs(%83 : memref<64xf16>)
            loom.semaphore_give %84 : memref<64xf16>
            %112 = loom.broadcast ins(%26 : memref<64xf16>) outs(%41 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            %113 = loom.broadcast ins(%83 : memref<64xf16>) outs(%86 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %83 : memref<64xf16>
            %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%111], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            loom.copy %reinterpret_cast_7, %82 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%82 : memref<64xf16>) outs(%81 : memref<64xf16>)
            loom.semaphore_give %82 : memref<64xf16>
            %114 = loom.broadcast ins(%81 : memref<64xf16>) outs(%88 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
            loom.semaphore_give %81 : memref<64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%78, %112, %113, %114 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%78 : memref<64x64xf16>) {
            ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
              %122 = arith.subf %in_9, %in_10 : f16
              %123 = math.exp %122 : f16
              %124 = arith.mulf %in, %123 : f16
              %125 = arith.mulf %124, %in_11 : f16
              linalg.yield %125 : f16
            }
            loom.semaphore_give %88 : memref<32x64xf16>
            loom.semaphore_give %86 : memref<32x64xf16>
            loom.semaphore_give %41 : memref<64x32xf16>
            %115 = arith.addi %102, %30 : index
            %116 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %117 = arith.muli %115, %c4096 overflow<nsw> : index
            %118 = arith.addi %116, %117 : index
            %119 = arith.muli %21, %c512 : index
            %120 = arith.addi %118, %119 : index
            %121 = arith.addi %120, %53 : index
            %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%121], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_8, %91 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%91 : memref<64x32xf16>) outs(%90 : memref<64x32xf16>)
            loom.semaphore_give %91 : memref<64x32xf16>
            loom.matmul ins(%78, %90 : memref<64x64xf16>, memref<64x32xf16>) outs(%76 : memref<64x32xf16>)
            loom.semaphore_give %90 : memref<64x32xf16>
            loom.semaphore_give %78 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%68, %76 : memref<64x32xf16>, memref<64x32xf16>) outs(%68 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_9: f16, %out: f16):
              %122 = arith.addf %in, %in_9 : f16
              linalg.yield %122 : f16
            }
            loom.semaphore_give %76 : memref<64x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %26 : memref<64xf16>
          %92 = loom.alloc [1] on @L1 : memref<f16>
          %93 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
          %94 = loom.semaphore_take %92 : memref<f16> -> memref<f16>
          %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
          %95 = arith.addi %36, %c3 : index
          loom.copy %reinterpret_cast_2, %94 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %36], LR : [%c7, %95]) : memref<f16, strided<[], offset: ?>> to memref<f16>
          loom.sync ins(%94 : memref<f16>) outs(%93 : memref<f16>)
          loom.semaphore_give %94 : memref<f16>
          %96 = arith.muli %arg11, %c8388608 overflow<nsw> : index
          %97 = arith.muli %44, %c4096 overflow<nsw> : index
          %98 = arith.addi %96, %97 : index
          %99 = arith.muli %21, %c512 : index
          %100 = arith.addi %98, %99 : index
          %101 = arith.addi %100, %53 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%101], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %75 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
          loom.sync ins(%75 : memref<64x32xf16>) outs(%74 : memref<64x32xf16>)
          loom.semaphore_give %75 : memref<64x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%68, %74, %93 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%74 : memref<64x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %102 = arith.mulf %in_5, %in_6 : f16
            %103 = arith.addf %in, %102 : f16
            linalg.yield %103 : f16
          }
          loom.semaphore_give %93 : memref<f16>
          loom.semaphore_give %68 : memref<64x32xf16>
          loom.sync ins(%74 : memref<64x32xf16>) outs(%73 : memref<64x32xf16>)
          loom.semaphore_give %74 : memref<64x32xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%101], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.copy %73, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%34, %65], LR : [%34, %65]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
          loom.semaphore_give %73 : memref<64x32xf16>
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_y, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c2, %c4) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %29 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %30 = arith.muli %21, %c16384 : index
            %31 = arith.addi %29, %30 : index
            %32 = arith.muli %arg12, %c1024 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %35 = arith.muli %arg11, %c2 : index
            %36 = arith.addi %arg9, %35 : index
            %37 = arith.muli %arg8, %c4 : index
            %38 = arith.addi %36, %37 : index
            %39 = arith.muli %arg12, %c2 : index
            %40 = arith.addi %39, %c1 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%38, %39], LR : [%38, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%27 : memref<64xf16>) outs(%28 : memref<64xf16>)
            loom.semaphore_give %27 : memref<64xf16>
            %41 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %42 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.broadcast ins(%28 : memref<64xf16>) outs(%43 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %45 = arith.addi %25, %32 : index
            %46 = arith.divui %24, %c64 : index
            %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %49 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %50 = arith.muli %45, %c64 overflow<nsw> : index
            %51 = arith.addi %29, %50 : index
            %52 = arith.muli %46, %c64 overflow<nsw> : index
            %53 = arith.addi %51, %52 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%53], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%38, %39], LR : [%38, %40]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            loom.sync ins(%49 : memref<64x64xf16>) outs(%48 : memref<64x64xf16>)
            loom.semaphore_give %49 : memref<64x64xf16>
            %54 = arith.muli %arg10, %c32 : index
            %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
            %57 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
            %58 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %59 = arith.muli %arg12, %c1048576 : index
            %60 = arith.addi %58, %59 : index
            %61 = arith.muli %21, %c32768 : index
            %62 = arith.addi %60, %61 : index
            %63 = arith.addi %62, %54 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %64 = arith.addi %35, %37 : index
            %65 = arith.addi %35, %c1 : index
            %66 = arith.addi %65, %37 : index
            %67 = arith.addi %arg10, %39 : index
            loom.copy %reinterpret_cast_1, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %67], LR : [%66, %67]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%57 : memref<64x32xf16>) outs(%56 : memref<64x32xf16>)
            loom.semaphore_give %57 : memref<64x32xf16>
            %68 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %69 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            %70 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%48, %56 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %56 : memref<64x32xf16>
            loom.semaphore_give %48 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %44 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%70 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %104 = math.exp %in_5 : f16
              %105 = arith.mulf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
            loom.semaphore_give %43 : memref<64x32xf16>
            %71 = arith.addi %23, %c1 : index
            %72 = arith.muli %71, %c64 : index
            %73 = arith.ceildivui %72, %c64 : index
            %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %76 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %77 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %78 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %79 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %80 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %81 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %82 = loom.alloc [64] on @L1 : memref<64xf16>
            %83 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %84 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %85 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %86 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %87 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %88 = loom.semaphore_take %87 : memref<32x64xf16> -> memref<32x64xf16>
            %89 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %90 = loom.semaphore_take %89 : memref<32x64xf16> -> memref<32x64xf16>
            %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            %93 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %73 step %c1 {
              %104 = arith.muli %arg15, %c64 : index
              %105 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %106 = arith.muli %arg12, %c262144 : index
              %107 = arith.addi %105, %106 : index
              %108 = arith.muli %46, %c65536 overflow<nsw> : index
              %109 = arith.addi %107, %108 : index
              %110 = arith.muli %23, %c16384 : index
              %111 = arith.addi %109, %110 : index
              %112 = arith.addi %111, %104 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%112], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              loom.sync ins(%81 : memref<64x64xf16>) outs(%80 : memref<64x64xf16>)
              loom.semaphore_give %81 : memref<64x64xf16>
              %113 = arith.addi %33, %104 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%86 : memref<64xf16>) outs(%85 : memref<64xf16>)
              loom.semaphore_give %86 : memref<64xf16>
              %114 = loom.broadcast ins(%28 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %115 = loom.broadcast ins(%85 : memref<64xf16>) outs(%88 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %85 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%84 : memref<64xf16>) outs(%83 : memref<64xf16>)
              loom.semaphore_give %84 : memref<64xf16>
              %116 = loom.broadcast ins(%83 : memref<64xf16>) outs(%90 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %83 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80, %114, %115, %116 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%80 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %124 = arith.subf %in_9, %in_10 : f16
                %125 = math.exp %124 : f16
                %126 = arith.mulf %in, %125 : f16
                %127 = arith.mulf %126, %in_11 : f16
                linalg.yield %127 : f16
              }
              loom.semaphore_give %90 : memref<32x64xf16>
              loom.semaphore_give %88 : memref<32x64xf16>
              loom.semaphore_give %42 : memref<64x32xf16>
              %117 = arith.addi %104, %32 : index
              %118 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %119 = arith.muli %117, %c4096 overflow<nsw> : index
              %120 = arith.addi %118, %119 : index
              %121 = arith.muli %21, %c512 : index
              %122 = arith.addi %120, %121 : index
              %123 = arith.addi %122, %54 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%123], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              loom.sync ins(%93 : memref<64x32xf16>) outs(%92 : memref<64x32xf16>)
              loom.semaphore_give %93 : memref<64x32xf16>
              loom.matmul ins(%80, %92 : memref<64x64xf16>, memref<64x32xf16>) outs(%78 : memref<64x32xf16>)
              loom.semaphore_give %92 : memref<64x32xf16>
              loom.semaphore_give %80 : memref<64x64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %78 : memref<64x32xf16>, memref<64x32xf16>) outs(%70 : memref<64x32xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.addf %in, %in_9 : f16
                linalg.yield %124 : f16
              }
              loom.semaphore_give %78 : memref<64x32xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %28 : memref<64xf16>
            %94 = loom.alloc [1] on @L1 : memref<f16>
            %95 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %96 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %97 = arith.addi %37, %c3 : index
            loom.copy %reinterpret_cast_2, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%37, %c0], LR : [%97, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            loom.sync ins(%96 : memref<f16>) outs(%95 : memref<f16>)
            loom.semaphore_give %96 : memref<f16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %45, %c4096 overflow<nsw> : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %21, %c512 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %54 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%77 : memref<64x32xf16>) outs(%76 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %76, %95 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%76 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %104 = arith.mulf %in_5, %in_6 : f16
              %105 = arith.addf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %95 : memref<f16>
            loom.semaphore_give %70 : memref<64x32xf16>
            loom.sync ins(%76 : memref<64x32xf16>) outs(%75 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %75, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %75 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y2y4__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc2_dim_y_level0_bc2_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c2, %c4, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %29 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %30 = arith.muli %21, %c16384 : index
            %31 = arith.addi %29, %30 : index
            %32 = arith.muli %arg12, %c1024 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %35 = arith.muli %arg12, %c2 : index
            %36 = arith.addi %arg9, %35 : index
            %37 = arith.muli %arg8, %c4 : index
            %38 = arith.addi %36, %37 : index
            %39 = arith.muli %arg11, %c2 : index
            %40 = arith.addi %39, %c1 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%38, %39], LR : [%38, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%27 : memref<64xf16>) outs(%28 : memref<64xf16>)
            loom.semaphore_give %27 : memref<64xf16>
            %41 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %42 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.broadcast ins(%28 : memref<64xf16>) outs(%43 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %45 = arith.addi %25, %32 : index
            %46 = arith.divui %24, %c64 : index
            %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %49 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %50 = arith.muli %45, %c64 overflow<nsw> : index
            %51 = arith.addi %29, %50 : index
            %52 = arith.muli %46, %c64 overflow<nsw> : index
            %53 = arith.addi %51, %52 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%53], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 2] region : (UL : [%38, %39], LR : [%38, %40]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            loom.sync ins(%49 : memref<64x64xf16>) outs(%48 : memref<64x64xf16>)
            loom.semaphore_give %49 : memref<64x64xf16>
            %54 = arith.muli %arg10, %c32 : index
            %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
            %57 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
            %58 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %59 = arith.muli %arg12, %c1048576 : index
            %60 = arith.addi %58, %59 : index
            %61 = arith.muli %21, %c32768 : index
            %62 = arith.addi %60, %61 : index
            %63 = arith.addi %62, %54 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %64 = arith.addi %35, %37 : index
            %65 = arith.addi %35, %c1 : index
            %66 = arith.addi %65, %37 : index
            %67 = arith.addi %arg10, %39 : index
            loom.copy %reinterpret_cast_1, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %67], LR : [%66, %67]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%57 : memref<64x32xf16>) outs(%56 : memref<64x32xf16>)
            loom.semaphore_give %57 : memref<64x32xf16>
            %68 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %69 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            %70 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%48, %56 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %56 : memref<64x32xf16>
            loom.semaphore_give %48 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %44 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%70 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %104 = math.exp %in_5 : f16
              %105 = arith.mulf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
            loom.semaphore_give %43 : memref<64x32xf16>
            %71 = arith.addi %23, %c1 : index
            %72 = arith.muli %71, %c64 : index
            %73 = arith.ceildivui %72, %c64 : index
            %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %76 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %77 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %78 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %79 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %80 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %81 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %82 = loom.alloc [64] on @L1 : memref<64xf16>
            %83 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %84 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %85 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %86 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %87 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %88 = loom.semaphore_take %87 : memref<32x64xf16> -> memref<32x64xf16>
            %89 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %90 = loom.semaphore_take %89 : memref<32x64xf16> -> memref<32x64xf16>
            %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            %93 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %73 step %c1 {
              %104 = arith.muli %arg15, %c64 : index
              %105 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %106 = arith.muli %arg12, %c262144 : index
              %107 = arith.addi %105, %106 : index
              %108 = arith.muli %46, %c65536 overflow<nsw> : index
              %109 = arith.addi %107, %108 : index
              %110 = arith.muli %23, %c16384 : index
              %111 = arith.addi %109, %110 : index
              %112 = arith.addi %111, %104 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%112], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              loom.sync ins(%81 : memref<64x64xf16>) outs(%80 : memref<64x64xf16>)
              loom.semaphore_give %81 : memref<64x64xf16>
              %113 = arith.addi %33, %104 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%86 : memref<64xf16>) outs(%85 : memref<64xf16>)
              loom.semaphore_give %86 : memref<64xf16>
              %114 = loom.broadcast ins(%28 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %115 = loom.broadcast ins(%85 : memref<64xf16>) outs(%88 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %85 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%84 : memref<64xf16>) outs(%83 : memref<64xf16>)
              loom.semaphore_give %84 : memref<64xf16>
              %116 = loom.broadcast ins(%83 : memref<64xf16>) outs(%90 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %83 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80, %114, %115, %116 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%80 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %124 = arith.subf %in_9, %in_10 : f16
                %125 = math.exp %124 : f16
                %126 = arith.mulf %in, %125 : f16
                %127 = arith.mulf %126, %in_11 : f16
                linalg.yield %127 : f16
              }
              loom.semaphore_give %90 : memref<32x64xf16>
              loom.semaphore_give %88 : memref<32x64xf16>
              loom.semaphore_give %42 : memref<64x32xf16>
              %117 = arith.addi %104, %32 : index
              %118 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %119 = arith.muli %117, %c4096 overflow<nsw> : index
              %120 = arith.addi %118, %119 : index
              %121 = arith.muli %21, %c512 : index
              %122 = arith.addi %120, %121 : index
              %123 = arith.addi %122, %54 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%123], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              loom.sync ins(%93 : memref<64x32xf16>) outs(%92 : memref<64x32xf16>)
              loom.semaphore_give %93 : memref<64x32xf16>
              loom.matmul ins(%80, %92 : memref<64x64xf16>, memref<64x32xf16>) outs(%78 : memref<64x32xf16>)
              loom.semaphore_give %92 : memref<64x32xf16>
              loom.semaphore_give %80 : memref<64x64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %78 : memref<64x32xf16>, memref<64x32xf16>) outs(%70 : memref<64x32xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.addf %in, %in_9 : f16
                linalg.yield %124 : f16
              }
              loom.semaphore_give %78 : memref<64x32xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %28 : memref<64xf16>
            %94 = loom.alloc [1] on @L1 : memref<f16>
            %95 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %96 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %97 = arith.addi %37, %c3 : index
            loom.copy %reinterpret_cast_2, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%37, %c0], LR : [%97, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            loom.sync ins(%96 : memref<f16>) outs(%95 : memref<f16>)
            loom.semaphore_give %96 : memref<f16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %45, %c4096 overflow<nsw> : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %21, %c512 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %54 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%77 : memref<64x32xf16>) outs(%76 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %76, %95 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%76 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %104 = arith.mulf %in_5, %in_6 : f16
              %105 = arith.addf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %95 : memref<f16>
            loom.semaphore_give %70 : memref<64x32xf16>
            loom.sync ins(%76 : memref<64x32xf16>) outs(%75 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %75, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %75 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i3_d3i4_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %29 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %30 = arith.muli %21, %c16384 : index
            %31 = arith.addi %29, %30 : index
            %32 = arith.muli %arg12, %c1024 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %35 = arith.muli %arg11, %c2 : index
            %36 = arith.addi %arg9, %35 : index
            %37 = arith.muli %arg8, %c4 : index
            %38 = arith.addi %36, %37 : index
            %39 = arith.muli %arg12, %c4 : index
            %40 = arith.addi %39, %c3 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%38, %39], LR : [%38, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%27 : memref<64xf16>) outs(%28 : memref<64xf16>)
            loom.semaphore_give %27 : memref<64xf16>
            %41 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %42 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.broadcast ins(%28 : memref<64xf16>) outs(%43 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %45 = arith.addi %25, %32 : index
            %46 = arith.divui %24, %c64 : index
            %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %49 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %50 = arith.muli %45, %c64 overflow<nsw> : index
            %51 = arith.addi %29, %50 : index
            %52 = arith.muli %46, %c64 overflow<nsw> : index
            %53 = arith.addi %51, %52 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%53], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%38, %39], LR : [%38, %40]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            loom.sync ins(%49 : memref<64x64xf16>) outs(%48 : memref<64x64xf16>)
            loom.semaphore_give %49 : memref<64x64xf16>
            %54 = arith.muli %arg10, %c32 : index
            %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
            %57 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
            %58 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %59 = arith.muli %arg12, %c1048576 : index
            %60 = arith.addi %58, %59 : index
            %61 = arith.muli %21, %c32768 : index
            %62 = arith.addi %60, %61 : index
            %63 = arith.addi %62, %54 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %64 = arith.addi %35, %37 : index
            %65 = arith.addi %35, %c1 : index
            %66 = arith.addi %65, %37 : index
            %67 = arith.addi %arg10, %39 : index
            loom.copy %reinterpret_cast_1, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %67], LR : [%66, %67]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%57 : memref<64x32xf16>) outs(%56 : memref<64x32xf16>)
            loom.semaphore_give %57 : memref<64x32xf16>
            %68 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %69 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            %70 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%48, %56 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %56 : memref<64x32xf16>
            loom.semaphore_give %48 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %44 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%70 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %104 = math.exp %in_5 : f16
              %105 = arith.mulf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
            loom.semaphore_give %43 : memref<64x32xf16>
            %71 = arith.addi %23, %c1 : index
            %72 = arith.muli %71, %c64 : index
            %73 = arith.ceildivui %72, %c64 : index
            %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %76 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %77 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %78 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %79 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %80 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %81 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %82 = loom.alloc [64] on @L1 : memref<64xf16>
            %83 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %84 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %85 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %86 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %87 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %88 = loom.semaphore_take %87 : memref<32x64xf16> -> memref<32x64xf16>
            %89 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %90 = loom.semaphore_take %89 : memref<32x64xf16> -> memref<32x64xf16>
            %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            %93 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %73 step %c1 {
              %104 = arith.muli %arg15, %c64 : index
              %105 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %106 = arith.muli %arg12, %c262144 : index
              %107 = arith.addi %105, %106 : index
              %108 = arith.muli %46, %c65536 overflow<nsw> : index
              %109 = arith.addi %107, %108 : index
              %110 = arith.muli %23, %c16384 : index
              %111 = arith.addi %109, %110 : index
              %112 = arith.addi %111, %104 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%112], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              loom.sync ins(%81 : memref<64x64xf16>) outs(%80 : memref<64x64xf16>)
              loom.semaphore_give %81 : memref<64x64xf16>
              %113 = arith.addi %33, %104 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%86 : memref<64xf16>) outs(%85 : memref<64xf16>)
              loom.semaphore_give %86 : memref<64xf16>
              %114 = loom.broadcast ins(%28 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %115 = loom.broadcast ins(%85 : memref<64xf16>) outs(%88 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %85 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%84 : memref<64xf16>) outs(%83 : memref<64xf16>)
              loom.semaphore_give %84 : memref<64xf16>
              %116 = loom.broadcast ins(%83 : memref<64xf16>) outs(%90 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %83 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80, %114, %115, %116 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%80 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %124 = arith.subf %in_9, %in_10 : f16
                %125 = math.exp %124 : f16
                %126 = arith.mulf %in, %125 : f16
                %127 = arith.mulf %126, %in_11 : f16
                linalg.yield %127 : f16
              }
              loom.semaphore_give %90 : memref<32x64xf16>
              loom.semaphore_give %88 : memref<32x64xf16>
              loom.semaphore_give %42 : memref<64x32xf16>
              %117 = arith.addi %104, %32 : index
              %118 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %119 = arith.muli %117, %c4096 overflow<nsw> : index
              %120 = arith.addi %118, %119 : index
              %121 = arith.muli %21, %c512 : index
              %122 = arith.addi %120, %121 : index
              %123 = arith.addi %122, %54 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%123], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              loom.sync ins(%93 : memref<64x32xf16>) outs(%92 : memref<64x32xf16>)
              loom.semaphore_give %93 : memref<64x32xf16>
              loom.matmul ins(%80, %92 : memref<64x64xf16>, memref<64x32xf16>) outs(%78 : memref<64x32xf16>)
              loom.semaphore_give %92 : memref<64x32xf16>
              loom.semaphore_give %80 : memref<64x64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %78 : memref<64x32xf16>, memref<64x32xf16>) outs(%70 : memref<64x32xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.addf %in, %in_9 : f16
                linalg.yield %124 : f16
              }
              loom.semaphore_give %78 : memref<64x32xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %28 : memref<64xf16>
            %94 = loom.alloc [1] on @L1 : memref<f16>
            %95 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %96 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %97 = arith.addi %37, %c3 : index
            loom.copy %reinterpret_cast_2, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%37, %c0], LR : [%97, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            loom.sync ins(%96 : memref<f16>) outs(%95 : memref<f16>)
            loom.semaphore_give %96 : memref<f16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %45, %c4096 overflow<nsw> : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %21, %c512 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %54 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%77 : memref<64x32xf16>) outs(%76 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %76, %95 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%76 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %104 = arith.mulf %in_5, %in_6 : f16
              %105 = arith.addf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %95 : memref<f16>
            loom.semaphore_give %70 : memref<64x32xf16>
            loom.sync ins(%76 : memref<64x32xf16>) outs(%75 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %75, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %75 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 2 : index}, loom.tile_c = {is_reduction = false, upper_bound = 8 : index}, loom.tile_h = {is_reduction = false, upper_bound = 64 : index}, loom.tile_k = {is_reduction = false, upper_bound = 8192 : index}, loom.tile_m = {is_reduction = false, upper_bound = 256 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}} {
    func.func @helion_mamba2_chunk_scan_kernel__x2x2x2_y4y2__d0i1_d1i2_d2i4_d3i3_d4i0__f01234__dim_y_level0_bc4_dim_y_level0_bc4_dim_x_level0_bc2_n_n_n_n_dim_x_level1_bc4_dim_y_level1_bc8_n_n__tile_b1__tile_c4__tile_h8__tile_k64__tile_m64__tile_n32(%arg0: memref<2x8x1x256x256xf16>, %arg1: memref<2x64x8x256xf16>, %arg2: memref<2x64x8x256xf16>, %arg3: memref<2x2048x64x64xf16>, %arg4: memref<2x2048x1x64xf16>, %arg5: memref<2x8x64x64x64xf16>, %arg6: memref<64xf16>, %arg7: memref<2x2048x64x64xf16>) {
      %c512 = arith.constant 512 : index
      %c262144 = arith.constant 262144 : index
      %c32768 = arith.constant 32768 : index
      %c1048576 = arith.constant 1048576 : index
      %c16384 = arith.constant 16384 : index
      %c8388608 = arith.constant 8388608 : index
      %c65536 = arith.constant 65536 : index
      %c524288 = arith.constant 524288 : index
      %c4096 = arith.constant 4096 : index
      %c2097152 = arith.constant 2097152 : index
      %c131072 = arith.constant 131072 : index
      %c1024 = arith.constant 1024 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.parallel (%arg8, %arg9, %arg10, %arg11, %arg12) = (%c0, %c0, %c0, %c0, %c0) to (%c2, %c2, %c4, %c2, %c2) step (%c1, %c1, %c1, %c1, %c1) {
        scf.for %arg13 = %c0 to %c4 step %c1 {
          scf.for %arg14 = %c0 to %c2 step %c1 {
            %20 = arith.muli %arg13, %c2 overflow<nsw> : index
            %21 = arith.addi %arg8, %20 : index
            %22 = arith.muli %arg14, %c2 overflow<nsw> : index
            %23 = arith.addi %arg9, %22 : index
            %24 = arith.muli %21, %c8 : index
            %25 = arith.muli %23, %c64 : index
            %26 = loom.alloc [64] on @L1 : memref<64xf16>
            %27 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %28 = loom.semaphore_take %26 : memref<64xf16> -> memref<64xf16>
            %29 = arith.muli %arg11, %c131072 overflow<nsw> : index
            %30 = arith.muli %21, %c16384 : index
            %31 = arith.addi %29, %30 : index
            %32 = arith.muli %arg12, %c1024 : index
            %33 = arith.addi %31, %32 : index
            %34 = arith.addi %33, %25 : index
            %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%34], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
            %35 = arith.muli %arg12, %c2 : index
            %36 = arith.addi %arg9, %35 : index
            %37 = arith.muli %arg8, %c4 : index
            %38 = arith.addi %36, %37 : index
            %39 = arith.muli %arg11, %c4 : index
            %40 = arith.addi %39, %c3 : index
            loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%38, %39], LR : [%38, %40]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
            loom.sync ins(%27 : memref<64xf16>) outs(%28 : memref<64xf16>)
            loom.semaphore_give %27 : memref<64xf16>
            %41 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %42 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %43 = loom.semaphore_take %41 : memref<64x32xf16> -> memref<64x32xf16>
            %44 = loom.broadcast ins(%28 : memref<64xf16>) outs(%43 : memref<64x32xf16>) dim(1) -> memref<64x32xf16, strided<[?, ?], offset: ?>>
            %45 = arith.addi %25, %32 : index
            %46 = arith.divui %24, %c64 : index
            %47 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %48 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %49 = loom.semaphore_take %47 : memref<64x64xf16> -> memref<64x64xf16>
            %50 = arith.muli %45, %c64 overflow<nsw> : index
            %51 = arith.addi %29, %50 : index
            %52 = arith.muli %46, %c64 overflow<nsw> : index
            %53 = arith.addi %51, %52 : index
            %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [%53], sizes: [64, 64], strides: [64, 1] : memref<2x2048x1x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            loom.copy %reinterpret_cast_0, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 4] region : (UL : [%38, %39], LR : [%38, %40]) : memref<64x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16>
            loom.sync ins(%49 : memref<64x64xf16>) outs(%48 : memref<64x64xf16>)
            loom.semaphore_give %49 : memref<64x64xf16>
            %54 = arith.muli %arg10, %c32 : index
            %55 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %56 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
            %57 = loom.semaphore_take %55 : memref<64x32xf16> -> memref<64x32xf16>
            %58 = arith.muli %arg11, %c2097152 overflow<nsw> : index
            %59 = arith.muli %arg12, %c1048576 : index
            %60 = arith.addi %58, %59 : index
            %61 = arith.muli %21, %c32768 : index
            %62 = arith.addi %60, %61 : index
            %63 = arith.addi %62, %54 : index
            %reinterpret_cast_1 = memref.reinterpret_cast %arg5 to offset: [%63], sizes: [64, 32], strides: [64, 1] : memref<2x8x64x64x64xf16> to memref<64x32xf16, strided<[64, 1], offset: ?>>
            %64 = arith.addi %35, %37 : index
            %65 = arith.addi %35, %c1 : index
            %66 = arith.addi %65, %37 : index
            %67 = arith.addi %arg10, %39 : index
            loom.copy %reinterpret_cast_1, %57 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 1] region : (UL : [%64, %67], LR : [%66, %67]) : memref<64x32xf16, strided<[64, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%57 : memref<64x32xf16>) outs(%56 : memref<64x32xf16>)
            loom.semaphore_give %57 : memref<64x32xf16>
            %68 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %69 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            %70 = loom.semaphore_take %68 : memref<64x32xf16> -> memref<64x32xf16>
            loom.matmul ins(%48, %56 : memref<64x64xf16>, memref<64x32xf16>) outs(%69 : memref<64x32xf16>)
            loom.semaphore_give %56 : memref<64x32xf16>
            loom.semaphore_give %48 : memref<64x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%69, %44 : memref<64x32xf16>, memref<64x32xf16, strided<[?, ?], offset: ?>>) outs(%70 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %104 = math.exp %in_5 : f16
              %105 = arith.mulf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %69 : memref<64x32xf16>
            loom.semaphore_give %43 : memref<64x32xf16>
            %71 = arith.addi %23, %c1 : index
            %72 = arith.muli %71, %c64 : index
            %73 = arith.ceildivui %72, %c64 : index
            %74 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %75 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %76 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %77 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %78 = loom.semaphore_take %74 : memref<64x32xf16> -> memref<64x32xf16>
            %79 = loom.alloc [64, 64] on @L1 : memref<64x64xf16>
            %80 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %81 = loom.semaphore_take %79 : memref<64x64xf16> -> memref<64x64xf16>
            %82 = loom.alloc [64] on @L1 : memref<64xf16>
            %83 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %84 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %85 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %86 = loom.semaphore_take %82 : memref<64xf16> -> memref<64xf16>
            %87 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %88 = loom.semaphore_take %87 : memref<32x64xf16> -> memref<32x64xf16>
            %89 = loom.alloc [32, 64] on @L1 : memref<32x64xf16>
            %90 = loom.semaphore_take %89 : memref<32x64xf16> -> memref<32x64xf16>
            %91 = loom.alloc [64, 32] on @L1 : memref<64x32xf16>
            %92 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            %93 = loom.semaphore_take %91 : memref<64x32xf16> -> memref<64x32xf16>
            scf.for %arg15 = %c0 to %73 step %c1 {
              %104 = arith.muli %arg15, %c64 : index
              %105 = arith.muli %arg11, %c524288 overflow<nsw> : index
              %106 = arith.muli %arg12, %c262144 : index
              %107 = arith.addi %105, %106 : index
              %108 = arith.muli %46, %c65536 overflow<nsw> : index
              %109 = arith.addi %107, %108 : index
              %110 = arith.muli %23, %c16384 : index
              %111 = arith.addi %109, %110 : index
              %112 = arith.addi %111, %104 : index
              %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [%112], sizes: [64, 64], strides: [256, 1] : memref<2x8x1x256x256xf16> to memref<64x64xf16, strided<[256, 1], offset: ?>>
              loom.copy %reinterpret_cast_5, %81 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x64xf16, strided<[256, 1], offset: ?>> to memref<64x64xf16>
              loom.sync ins(%81 : memref<64x64xf16>) outs(%80 : memref<64x64xf16>)
              loom.semaphore_give %81 : memref<64x64xf16>
              %113 = arith.addi %33, %104 : index
              %reinterpret_cast_6 = memref.reinterpret_cast %arg1 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_6, %86 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%86 : memref<64xf16>) outs(%85 : memref<64xf16>)
              loom.semaphore_give %86 : memref<64xf16>
              %114 = loom.broadcast ins(%28 : memref<64xf16>) outs(%42 : memref<64x32xf16>) dim(1) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              %115 = loom.broadcast ins(%85 : memref<64xf16>) outs(%88 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %85 : memref<64xf16>
              %reinterpret_cast_7 = memref.reinterpret_cast %arg2 to offset: [%113], sizes: [64], strides: [1] : memref<2x64x8x256xf16> to memref<64xf16, strided<[1], offset: ?>>
              loom.copy %reinterpret_cast_7, %84 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64xf16, strided<[1], offset: ?>> to memref<64xf16>
              loom.sync ins(%84 : memref<64xf16>) outs(%83 : memref<64xf16>)
              loom.semaphore_give %84 : memref<64xf16>
              %116 = loom.broadcast ins(%83 : memref<64xf16>) outs(%90 : memref<32x64xf16>) dim(0) -> memref<64x64xf16, strided<[?, ?], offset: ?>>
              loom.semaphore_give %83 : memref<64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%80, %114, %115, %116 : memref<64x64xf16>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>, memref<64x64xf16, strided<[?, ?], offset: ?>>) outs(%80 : memref<64x64xf16>) {
              ^bb0(%in: f16, %in_9: f16, %in_10: f16, %in_11: f16, %out: f16):
                %124 = arith.subf %in_9, %in_10 : f16
                %125 = math.exp %124 : f16
                %126 = arith.mulf %in, %125 : f16
                %127 = arith.mulf %126, %in_11 : f16
                linalg.yield %127 : f16
              }
              loom.semaphore_give %90 : memref<32x64xf16>
              loom.semaphore_give %88 : memref<32x64xf16>
              loom.semaphore_give %42 : memref<64x32xf16>
              %117 = arith.addi %104, %32 : index
              %118 = arith.muli %arg11, %c8388608 overflow<nsw> : index
              %119 = arith.muli %117, %c4096 overflow<nsw> : index
              %120 = arith.addi %118, %119 : index
              %121 = arith.muli %21, %c512 : index
              %122 = arith.addi %120, %121 : index
              %123 = arith.addi %122, %54 : index
              %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%123], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
              loom.copy %reinterpret_cast_8, %93 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
              loom.sync ins(%93 : memref<64x32xf16>) outs(%92 : memref<64x32xf16>)
              loom.semaphore_give %93 : memref<64x32xf16>
              loom.matmul ins(%80, %92 : memref<64x64xf16>, memref<64x32xf16>) outs(%78 : memref<64x32xf16>)
              loom.semaphore_give %92 : memref<64x32xf16>
              loom.semaphore_give %80 : memref<64x64xf16>
              linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %78 : memref<64x32xf16>, memref<64x32xf16>) outs(%70 : memref<64x32xf16>) {
              ^bb0(%in: f16, %in_9: f16, %out: f16):
                %124 = arith.addf %in, %in_9 : f16
                linalg.yield %124 : f16
              }
              loom.semaphore_give %78 : memref<64x32xf16>
            } {loom.iter_type = #loom.iter_type<sequential>}
            loom.semaphore_give %28 : memref<64xf16>
            %94 = loom.alloc [1] on @L1 : memref<f16>
            %95 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %96 = loom.semaphore_take %94 : memref<f16> -> memref<f16>
            %reinterpret_cast_2 = memref.reinterpret_cast %arg6 to offset: [%24], sizes: [], strides: [] : memref<64xf16> to memref<f16, strided<[], offset: ?>>
            %97 = arith.addi %37, %c3 : index
            loom.copy %reinterpret_cast_2, %96 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%37, %c0], LR : [%97, %c7]) : memref<f16, strided<[], offset: ?>> to memref<f16>
            loom.sync ins(%96 : memref<f16>) outs(%95 : memref<f16>)
            loom.semaphore_give %96 : memref<f16>
            %98 = arith.muli %arg11, %c8388608 overflow<nsw> : index
            %99 = arith.muli %45, %c4096 overflow<nsw> : index
            %100 = arith.addi %98, %99 : index
            %101 = arith.muli %21, %c512 : index
            %102 = arith.addi %100, %101 : index
            %103 = arith.addi %102, %54 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %77 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16, strided<[4096, 1], offset: ?>> to memref<64x32xf16>
            loom.sync ins(%77 : memref<64x32xf16>) outs(%76 : memref<64x32xf16>)
            loom.semaphore_give %77 : memref<64x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%70, %76, %95 : memref<64x32xf16>, memref<64x32xf16>, memref<f16>) outs(%76 : memref<64x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %104 = arith.mulf %in_5, %in_6 : f16
              %105 = arith.addf %in, %104 : f16
              linalg.yield %105 : f16
            }
            loom.semaphore_give %95 : memref<f16>
            loom.semaphore_give %70 : memref<64x32xf16>
            loom.sync ins(%76 : memref<64x32xf16>) outs(%75 : memref<64x32xf16>)
            loom.semaphore_give %76 : memref<64x32xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg7 to offset: [%103], sizes: [64, 32], strides: [4096, 1] : memref<2x2048x64x64xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.copy %75, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%38, %67], LR : [%38, %67]) : memref<64x32xf16> to memref<64x32xf16, strided<[4096, 1], offset: ?>>
            loom.semaphore_give %75 : memref<64x32xf16>
          } {loom.block_sym = @tile_m, loom.iter_type = #loom.iter_type<temporal>}
        } {loom.block_sym = @tile_h, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [2, 0, 0, 1, 1], loom.physical_dims = [@dim_x, @dim_x, @dim_y, @dim_y, @dim_x]}
      return
    }
  }
}
