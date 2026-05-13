module attributes {loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y1y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_x_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8192 = arith.constant 8192 : index
      %c131072 = arith.constant 131072 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %24 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %25 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %26 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %27 = arith.muli %21, %c524288 overflow<nsw> : index
          %28 = arith.muli %arg5, %c131072 : index
          %29 = arith.addi %27, %28 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
          loom.sync ins(%25 : memref<1x1024x128xf16>) outs(%26 : memref<1x1024x128xf16>)
          loom.semaphore_give %25 : memref<1x1024x128xf16>
          %30 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %31 = loom.semaphore_take %30 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          linalg.fill ins(%cst : f16) outs(%31 : memref<1x1024x128xf16>)
          %32 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%33 : memref<1x1024x1xf16>)
          %34 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %35 = loom.semaphore_take %34 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x1024x1xf16>)
          %36 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %37 = loom.semaphore_take %36 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %40 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %44 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %45 = loom.semaphore_take %44 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %46 = loom.semaphore_take %44 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %47 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
          %48 = loom.semaphore_take %47 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
          %49 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
          %50 = loom.semaphore_take %49 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %51 = loom.semaphore_take %49 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %52 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %53 = loom.semaphore_take %52 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          %54 = loom.semaphore_take %52 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg7 = %c0 to %c64 step %c1 {
            %56 = arith.muli %arg7, %c64 : index
            %57 = arith.addi %27, %56 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
            loom.sync ins(%46 : memref<1x128x64xf16>) outs(%45 : memref<1x128x64xf16>)
            loom.semaphore_give %46 : memref<1x128x64xf16>
            linalg.fill ins(%cst : f16) outs(%48 : memref<1x1024x64xf16>)
            linalg.batch_matmul ins(%26, %45 : memref<1x1024x128xf16>, memref<1x128x64xf16>) outs(%48 : memref<1x1024x64xf16>)
            loom.semaphore_give %45 : memref<1x128x64xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x1024x64xf16>) outs(%39 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.maximumf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%39 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in_6, %cst_2 : f16
              %62 = arith.cmpf ogt, %in, %61 : f16
              %63 = arith.select %62, %in, %61 : f16
              linalg.yield %63 : f16
            }
            %58 = loom.broadcast ins(%39 : memref<1x1024x1xf16>) outs(%51 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %58 : memref<1x1024x64xf16>, memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<1x1024x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %cst_2 : f16
              %62 = arith.subf %61, %in_6 : f16
              %63 = math.exp %62 : f16
              linalg.yield %63 : f16
            }
            loom.semaphore_give %51 : memref<1x1024x32xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x1024x64xf16>) outs(%41 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.addf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %43, %41 : memref<1x1024x1xf16>, memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%33 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              %62 = arith.addf %61, %in_7 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %41 : memref<1x1024x1xf16>
            %59 = arith.muli %arg7, %c8192 : index
            %60 = arith.addi %27, %59 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.sync ins(%54 : memref<1x64x128xf16>) outs(%53 : memref<1x64x128xf16>)
            loom.semaphore_give %54 : memref<1x64x128xf16>
            linalg.fill ins(%cst : f16) outs(%37 : memref<1x1024x128xf16>)
            linalg.batch_matmul ins(%48, %53 : memref<1x1024x64xf16>, memref<1x64x128xf16>) outs(%37 : memref<1x1024x128xf16>)
            loom.semaphore_give %53 : memref<1x64x128xf16>
            loom.semaphore_give %48 : memref<1x1024x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %31, %43 : memref<1x1024x128xf16>, memref<1x1024x128xf16>, memref<1x1024x1xf16>) outs(%31 : memref<1x1024x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in_6, %in_7 : f16
              %62 = arith.addf %in, %61 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %43 : memref<1x1024x1xf16>
            loom.semaphore_give %37 : memref<1x1024x128xf16>
            linalg.copy ins(%39 : memref<1x1024x1xf16>) outs(%35 : memref<1x1024x1xf16>)
            loom.semaphore_give %39 : memref<1x1024x1xf16>
          }
          loom.semaphore_give %35 : memref<1x1024x1xf16>
          loom.semaphore_give %26 : memref<1x1024x128xf16>
          %55 = loom.broadcast ins(%33 : memref<1x1024x1xf16>) outs(%50 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %33 : memref<1x1024x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %55 : memref<1x1024x128xf16>, memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%24 : memref<1x1024x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %56 = arith.divf %in, %in_4 : f16
            linalg.yield %56 : f16
          }
          loom.semaphore_give %50 : memref<1x1024x32xf16>
          loom.semaphore_give %31 : memref<1x1024x128xf16>
          loom.sync ins(%24 : memref<1x1024x128xf16>) outs(%23 : memref<1x1024x128xf16>)
          loom.semaphore_give %24 : memref<1x1024x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<1x1024x128xf16>
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      }
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y2y4__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc2_dim_x_level0_bc8_dim_y_level0_bc2_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8192 = arith.constant 8192 : index
      %c131072 = arith.constant 131072 : index
      %c524288 = arith.constant 524288 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c4, %c8, %c2) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c8 step %c1 {
          %20 = arith.muli %arg7, %c4 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = arith.muli %arg5, %c8 overflow<nsw> : index
          %23 = arith.addi %22, %arg6 : index
          %24 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %26 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %27 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %28 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %29 = arith.muli %21, %c524288 overflow<nsw> : index
          %30 = arith.muli %23, %c131072 : index
          %31 = arith.addi %29, %30 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          %32 = arith.muli %arg4, %c2 : index
          %33 = arith.addi %arg6, %32 : index
          loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
          loom.sync ins(%27 : memref<1x1024x128xf16>) outs(%28 : memref<1x1024x128xf16>)
          loom.semaphore_give %27 : memref<1x1024x128xf16>
          %34 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %35 = loom.semaphore_take %34 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          linalg.fill ins(%cst : f16) outs(%35 : memref<1x1024x128xf16>)
          %36 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%37 : memref<1x1024x1xf16>)
          %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x1024x1xf16>)
          %40 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %41 = loom.semaphore_take %40 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %44 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %45 = loom.semaphore_take %44 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %46 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %47 = loom.semaphore_take %46 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %48 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %49 = loom.semaphore_take %48 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %50 = loom.semaphore_take %48 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %51 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
          %52 = loom.semaphore_take %51 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
          %53 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
          %54 = loom.semaphore_take %53 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %55 = loom.semaphore_take %53 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %56 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %57 = loom.semaphore_take %56 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          %58 = loom.semaphore_take %56 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg8 = %c0 to %c64 step %c1 {
            %60 = arith.muli %arg8, %c64 : index
            %61 = arith.addi %29, %60 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%61], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
            %62 = arith.addi %32, %c1 : index
            loom.copy %reinterpret_cast_4, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 2] region : (UL : [%c0, %32], LR : [%c7, %62]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
            loom.sync ins(%50 : memref<1x128x64xf16>) outs(%49 : memref<1x128x64xf16>)
            loom.semaphore_give %50 : memref<1x128x64xf16>
            linalg.fill ins(%cst : f16) outs(%52 : memref<1x1024x64xf16>)
            linalg.batch_matmul ins(%28, %49 : memref<1x1024x128xf16>, memref<1x128x64xf16>) outs(%52 : memref<1x1024x64xf16>)
            loom.semaphore_give %49 : memref<1x128x64xf16>
            linalg.fill ins(%cst_1 : f16) outs(%43 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : memref<1x1024x64xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %66 = arith.maximumf %in, %out : f16
              linalg.yield %66 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %43 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.mulf %in_6, %cst_2 : f16
              %67 = arith.cmpf ogt, %in, %66 : f16
              %68 = arith.select %67, %in, %66 : f16
              linalg.yield %68 : f16
            }
            %63 = loom.broadcast ins(%43 : memref<1x1024x1xf16>) outs(%55 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52, %63 : memref<1x1024x64xf16>, memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>) outs(%52 : memref<1x1024x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.mulf %in, %cst_2 : f16
              %67 = arith.subf %66, %in_6 : f16
              %68 = math.exp %67 : f16
              linalg.yield %68 : f16
            }
            loom.semaphore_give %55 : memref<1x1024x32xf16>
            linalg.fill ins(%cst : f16) outs(%45 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : memref<1x1024x64xf16>) outs(%45 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %66 = arith.addf %in, %out : f16
              linalg.yield %66 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %43 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%47 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.subf %in, %in_6 : f16
              %67 = math.exp %66 : f16
              linalg.yield %67 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %47, %45 : memref<1x1024x1xf16>, memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%37 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %66 = arith.mulf %in, %in_6 : f16
              %67 = arith.addf %66, %in_7 : f16
              linalg.yield %67 : f16
            }
            loom.semaphore_give %45 : memref<1x1024x1xf16>
            %64 = arith.muli %arg8, %c8192 : index
            %65 = arith.addi %29, %64 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%65], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 2] region : (UL : [%c0, %32], LR : [%c7, %62]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.sync ins(%58 : memref<1x64x128xf16>) outs(%57 : memref<1x64x128xf16>)
            loom.semaphore_give %58 : memref<1x64x128xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x1024x128xf16>)
            linalg.batch_matmul ins(%52, %57 : memref<1x1024x64xf16>, memref<1x64x128xf16>) outs(%41 : memref<1x1024x128xf16>)
            loom.semaphore_give %57 : memref<1x64x128xf16>
            loom.semaphore_give %52 : memref<1x1024x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %35, %47 : memref<1x1024x128xf16>, memref<1x1024x128xf16>, memref<1x1024x1xf16>) outs(%35 : memref<1x1024x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %66 = arith.mulf %in_6, %in_7 : f16
              %67 = arith.addf %in, %66 : f16
              linalg.yield %67 : f16
            }
            loom.semaphore_give %47 : memref<1x1024x1xf16>
            loom.semaphore_give %41 : memref<1x1024x128xf16>
            linalg.copy ins(%43 : memref<1x1024x1xf16>) outs(%39 : memref<1x1024x1xf16>)
            loom.semaphore_give %43 : memref<1x1024x1xf16>
          }
          loom.semaphore_give %39 : memref<1x1024x1xf16>
          loom.semaphore_give %28 : memref<1x1024x128xf16>
          %59 = loom.broadcast ins(%37 : memref<1x1024x1xf16>) outs(%54 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %37 : memref<1x1024x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %59 : memref<1x1024x128xf16>, memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%26 : memref<1x1024x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %60 = arith.divf %in, %in_4 : f16
            linalg.yield %60 : f16
          }
          loom.semaphore_give %54 : memref<1x1024x32xf16>
          loom.semaphore_give %35 : memref<1x1024x128xf16>
          loom.sync ins(%26 : memref<1x1024x128xf16>) outs(%25 : memref<1x1024x128xf16>)
          loom.semaphore_give %26 : memref<1x1024x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %25, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %25 : memref<1x1024x128xf16>
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y4y2__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level0_bc4_dim_x_level0_bc8_dim_y_level0_bc4_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8192 = arith.constant 8192 : index
      %c131072 = arith.constant 131072 : index
      %c524288 = arith.constant 524288 : index
      %c16 = arith.constant 16 : index
      %c3 = arith.constant 3 : index
      %c7 = arith.constant 7 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c2, %c8, %c4) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg7, %c2 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = arith.muli %arg5, %c8 overflow<nsw> : index
          %23 = arith.addi %22, %arg6 : index
          %24 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %26 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %27 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %28 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %29 = arith.muli %21, %c524288 overflow<nsw> : index
          %30 = arith.muli %23, %c131072 : index
          %31 = arith.addi %29, %30 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          %32 = arith.muli %arg4, %c4 : index
          %33 = arith.addi %arg6, %32 : index
          loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
          loom.sync ins(%27 : memref<1x1024x128xf16>) outs(%28 : memref<1x1024x128xf16>)
          loom.semaphore_give %27 : memref<1x1024x128xf16>
          %34 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %35 = loom.semaphore_take %34 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          linalg.fill ins(%cst : f16) outs(%35 : memref<1x1024x128xf16>)
          %36 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%37 : memref<1x1024x1xf16>)
          %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x1024x1xf16>)
          %40 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %41 = loom.semaphore_take %40 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %44 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %45 = loom.semaphore_take %44 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %46 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %47 = loom.semaphore_take %46 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %48 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %49 = loom.semaphore_take %48 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %50 = loom.semaphore_take %48 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %51 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
          %52 = loom.semaphore_take %51 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
          %53 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
          %54 = loom.semaphore_take %53 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %55 = loom.semaphore_take %53 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %56 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %57 = loom.semaphore_take %56 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          %58 = loom.semaphore_take %56 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg8 = %c0 to %c64 step %c1 {
            %60 = arith.muli %arg8, %c64 : index
            %61 = arith.addi %29, %60 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%61], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
            %62 = arith.addi %32, %c3 : index
            loom.copy %reinterpret_cast_4, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %62]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
            loom.sync ins(%50 : memref<1x128x64xf16>) outs(%49 : memref<1x128x64xf16>)
            loom.semaphore_give %50 : memref<1x128x64xf16>
            linalg.fill ins(%cst : f16) outs(%52 : memref<1x1024x64xf16>)
            linalg.batch_matmul ins(%28, %49 : memref<1x1024x128xf16>, memref<1x128x64xf16>) outs(%52 : memref<1x1024x64xf16>)
            loom.semaphore_give %49 : memref<1x128x64xf16>
            linalg.fill ins(%cst_1 : f16) outs(%43 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : memref<1x1024x64xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %66 = arith.maximumf %in, %out : f16
              linalg.yield %66 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %43 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.mulf %in_6, %cst_2 : f16
              %67 = arith.cmpf ogt, %in, %66 : f16
              %68 = arith.select %67, %in, %66 : f16
              linalg.yield %68 : f16
            }
            %63 = loom.broadcast ins(%43 : memref<1x1024x1xf16>) outs(%55 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52, %63 : memref<1x1024x64xf16>, memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>) outs(%52 : memref<1x1024x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.mulf %in, %cst_2 : f16
              %67 = arith.subf %66, %in_6 : f16
              %68 = math.exp %67 : f16
              linalg.yield %68 : f16
            }
            loom.semaphore_give %55 : memref<1x1024x32xf16>
            linalg.fill ins(%cst : f16) outs(%45 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : memref<1x1024x64xf16>) outs(%45 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %66 = arith.addf %in, %out : f16
              linalg.yield %66 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %43 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%47 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.subf %in, %in_6 : f16
              %67 = math.exp %66 : f16
              linalg.yield %67 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %47, %45 : memref<1x1024x1xf16>, memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%37 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %66 = arith.mulf %in, %in_6 : f16
              %67 = arith.addf %66, %in_7 : f16
              linalg.yield %67 : f16
            }
            loom.semaphore_give %45 : memref<1x1024x1xf16>
            %64 = arith.muli %arg8, %c8192 : index
            %65 = arith.addi %29, %64 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%65], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 4] region : (UL : [%c0, %32], LR : [%c7, %62]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.sync ins(%58 : memref<1x64x128xf16>) outs(%57 : memref<1x64x128xf16>)
            loom.semaphore_give %58 : memref<1x64x128xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x1024x128xf16>)
            linalg.batch_matmul ins(%52, %57 : memref<1x1024x64xf16>, memref<1x64x128xf16>) outs(%41 : memref<1x1024x128xf16>)
            loom.semaphore_give %57 : memref<1x64x128xf16>
            loom.semaphore_give %52 : memref<1x1024x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %35, %47 : memref<1x1024x128xf16>, memref<1x1024x128xf16>, memref<1x1024x1xf16>) outs(%35 : memref<1x1024x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %66 = arith.mulf %in_6, %in_7 : f16
              %67 = arith.addf %in, %66 : f16
              linalg.yield %67 : f16
            }
            loom.semaphore_give %47 : memref<1x1024x1xf16>
            loom.semaphore_give %41 : memref<1x1024x128xf16>
            linalg.copy ins(%43 : memref<1x1024x1xf16>) outs(%39 : memref<1x1024x1xf16>)
            loom.semaphore_give %43 : memref<1x1024x1xf16>
          }
          loom.semaphore_give %39 : memref<1x1024x1xf16>
          loom.semaphore_give %28 : memref<1x1024x128xf16>
          %59 = loom.broadcast ins(%37 : memref<1x1024x1xf16>) outs(%54 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %37 : memref<1x1024x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %59 : memref<1x1024x128xf16>, memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%26 : memref<1x1024x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %60 = arith.divf %in, %in_4 : f16
            linalg.yield %60 : f16
          }
          loom.semaphore_give %54 : memref<1x1024x32xf16>
          loom.semaphore_give %35 : memref<1x1024x128xf16>
          loom.sync ins(%26 : memref<1x1024x128xf16>) outs(%25 : memref<1x1024x128xf16>)
          loom.semaphore_give %26 : memref<1x1024x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %25, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %33], LR : [%arg5, %33]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %25 : memref<1x1024x128xf16>
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_y, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8_y8y1__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc8_dim_y_level1_bc8_dim_x_level0_bc8_dim_y_level1_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8192 = arith.constant 8192 : index
      %c131072 = arith.constant 131072 : index
      %c524288 = arith.constant 524288 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %20 = arith.muli %arg4, %c8 overflow<nsw> : index
          %21 = arith.addi %20, %arg5 : index
          %22 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %24 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %25 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %26 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %27 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %28 = arith.muli %21, %c131072 : index
          %29 = arith.addi %27, %28 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
          loom.sync ins(%25 : memref<1x1024x128xf16>) outs(%26 : memref<1x1024x128xf16>)
          loom.semaphore_give %25 : memref<1x1024x128xf16>
          %30 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %31 = loom.semaphore_take %30 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          linalg.fill ins(%cst : f16) outs(%31 : memref<1x1024x128xf16>)
          %32 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%33 : memref<1x1024x1xf16>)
          %34 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %35 = loom.semaphore_take %34 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x1024x1xf16>)
          %36 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %37 = loom.semaphore_take %36 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %40 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %44 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %45 = loom.semaphore_take %44 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %46 = loom.semaphore_take %44 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %47 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
          %48 = loom.semaphore_take %47 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
          %49 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
          %50 = loom.semaphore_take %49 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %51 = loom.semaphore_take %49 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %52 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %53 = loom.semaphore_take %52 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          %54 = loom.semaphore_take %52 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg7 = %c0 to %c64 step %c1 {
            %56 = arith.muli %arg7, %c64 : index
            %57 = arith.addi %27, %56 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
            loom.sync ins(%46 : memref<1x128x64xf16>) outs(%45 : memref<1x128x64xf16>)
            loom.semaphore_give %46 : memref<1x128x64xf16>
            linalg.fill ins(%cst : f16) outs(%48 : memref<1x1024x64xf16>)
            linalg.batch_matmul ins(%26, %45 : memref<1x1024x128xf16>, memref<1x128x64xf16>) outs(%48 : memref<1x1024x64xf16>)
            loom.semaphore_give %45 : memref<1x128x64xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x1024x64xf16>) outs(%39 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.maximumf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%39 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in_6, %cst_2 : f16
              %62 = arith.cmpf ogt, %in, %61 : f16
              %63 = arith.select %62, %in, %61 : f16
              linalg.yield %63 : f16
            }
            %58 = loom.broadcast ins(%39 : memref<1x1024x1xf16>) outs(%51 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %58 : memref<1x1024x64xf16>, memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<1x1024x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %cst_2 : f16
              %62 = arith.subf %61, %in_6 : f16
              %63 = math.exp %62 : f16
              linalg.yield %63 : f16
            }
            loom.semaphore_give %51 : memref<1x1024x32xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x1024x64xf16>) outs(%41 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.addf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %43, %41 : memref<1x1024x1xf16>, memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%33 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              %62 = arith.addf %61, %in_7 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %41 : memref<1x1024x1xf16>
            %59 = arith.muli %arg7, %c8192 : index
            %60 = arith.addi %27, %59 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.sync ins(%54 : memref<1x64x128xf16>) outs(%53 : memref<1x64x128xf16>)
            loom.semaphore_give %54 : memref<1x64x128xf16>
            linalg.fill ins(%cst : f16) outs(%37 : memref<1x1024x128xf16>)
            linalg.batch_matmul ins(%48, %53 : memref<1x1024x64xf16>, memref<1x64x128xf16>) outs(%37 : memref<1x1024x128xf16>)
            loom.semaphore_give %53 : memref<1x64x128xf16>
            loom.semaphore_give %48 : memref<1x1024x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %31, %43 : memref<1x1024x128xf16>, memref<1x1024x128xf16>, memref<1x1024x1xf16>) outs(%31 : memref<1x1024x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in_6, %in_7 : f16
              %62 = arith.addf %in, %61 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %43 : memref<1x1024x1xf16>
            loom.semaphore_give %37 : memref<1x1024x128xf16>
            linalg.copy ins(%39 : memref<1x1024x1xf16>) outs(%35 : memref<1x1024x1xf16>)
            loom.semaphore_give %39 : memref<1x1024x1xf16>
          }
          loom.semaphore_give %35 : memref<1x1024x1xf16>
          loom.semaphore_give %26 : memref<1x1024x128xf16>
          %55 = loom.broadcast ins(%33 : memref<1x1024x1xf16>) outs(%50 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %33 : memref<1x1024x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %55 : memref<1x1024x128xf16>, memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%24 : memref<1x1024x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %56 = arith.divf %in, %in_4 : f16
            linalg.yield %56 : f16
          }
          loom.semaphore_give %50 : memref<1x1024x32xf16>
          loom.semaphore_give %31 : memref<1x1024x128xf16>
          loom.sync ins(%24 : memref<1x1024x128xf16>) outs(%23 : memref<1x1024x128xf16>)
          loom.semaphore_give %24 : memref<1x1024x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<1x1024x128xf16>
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      }
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x1x8_y8__d0i1_d1i1_d2i0__f01__n_dim_y_level0_bc8_dim_y_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8192 = arith.constant 8192 : index
      %c131072 = arith.constant 131072 : index
      %c524288 = arith.constant 524288 : index
      %c4 = arith.constant 4 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c4 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %24 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %25 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %26 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %27 = arith.muli %21, %c524288 overflow<nsw> : index
          %28 = arith.muli %arg5, %c131072 : index
          %29 = arith.addi %27, %28 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
          loom.sync ins(%25 : memref<1x1024x128xf16>) outs(%26 : memref<1x1024x128xf16>)
          loom.semaphore_give %25 : memref<1x1024x128xf16>
          %30 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %31 = loom.semaphore_take %30 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          linalg.fill ins(%cst : f16) outs(%31 : memref<1x1024x128xf16>)
          %32 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%33 : memref<1x1024x1xf16>)
          %34 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %35 = loom.semaphore_take %34 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x1024x1xf16>)
          %36 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %37 = loom.semaphore_take %36 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %40 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %44 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %45 = loom.semaphore_take %44 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %46 = loom.semaphore_take %44 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %47 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
          %48 = loom.semaphore_take %47 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
          %49 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
          %50 = loom.semaphore_take %49 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %51 = loom.semaphore_take %49 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %52 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %53 = loom.semaphore_take %52 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          %54 = loom.semaphore_take %52 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg7 = %c0 to %c64 step %c1 {
            %56 = arith.muli %arg7, %c64 : index
            %57 = arith.addi %27, %56 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
            loom.sync ins(%46 : memref<1x128x64xf16>) outs(%45 : memref<1x128x64xf16>)
            loom.semaphore_give %46 : memref<1x128x64xf16>
            linalg.fill ins(%cst : f16) outs(%48 : memref<1x1024x64xf16>)
            linalg.batch_matmul ins(%26, %45 : memref<1x1024x128xf16>, memref<1x128x64xf16>) outs(%48 : memref<1x1024x64xf16>)
            loom.semaphore_give %45 : memref<1x128x64xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x1024x64xf16>) outs(%39 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.maximumf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%39 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in_6, %cst_2 : f16
              %62 = arith.cmpf ogt, %in, %61 : f16
              %63 = arith.select %62, %in, %61 : f16
              linalg.yield %63 : f16
            }
            %58 = loom.broadcast ins(%39 : memref<1x1024x1xf16>) outs(%51 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %58 : memref<1x1024x64xf16>, memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<1x1024x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %cst_2 : f16
              %62 = arith.subf %61, %in_6 : f16
              %63 = math.exp %62 : f16
              linalg.yield %63 : f16
            }
            loom.semaphore_give %51 : memref<1x1024x32xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x1024x64xf16>) outs(%41 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.addf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %43, %41 : memref<1x1024x1xf16>, memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%33 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              %62 = arith.addf %61, %in_7 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %41 : memref<1x1024x1xf16>
            %59 = arith.muli %arg7, %c8192 : index
            %60 = arith.addi %27, %59 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 8] region : (UL : [%arg4, %c0], LR : [%arg4, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.sync ins(%54 : memref<1x64x128xf16>) outs(%53 : memref<1x64x128xf16>)
            loom.semaphore_give %54 : memref<1x64x128xf16>
            linalg.fill ins(%cst : f16) outs(%37 : memref<1x1024x128xf16>)
            linalg.batch_matmul ins(%48, %53 : memref<1x1024x64xf16>, memref<1x64x128xf16>) outs(%37 : memref<1x1024x128xf16>)
            loom.semaphore_give %53 : memref<1x64x128xf16>
            loom.semaphore_give %48 : memref<1x1024x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %31, %43 : memref<1x1024x128xf16>, memref<1x1024x128xf16>, memref<1x1024x1xf16>) outs(%31 : memref<1x1024x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in_6, %in_7 : f16
              %62 = arith.addf %in, %61 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %43 : memref<1x1024x1xf16>
            loom.semaphore_give %37 : memref<1x1024x128xf16>
            linalg.copy ins(%39 : memref<1x1024x1xf16>) outs(%35 : memref<1x1024x1xf16>)
            loom.semaphore_give %39 : memref<1x1024x1xf16>
          }
          loom.semaphore_give %35 : memref<1x1024x1xf16>
          loom.semaphore_give %26 : memref<1x1024x128xf16>
          %55 = loom.broadcast ins(%33 : memref<1x1024x1xf16>) outs(%50 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %33 : memref<1x1024x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %55 : memref<1x1024x128xf16>, memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%24 : memref<1x1024x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %56 = arith.divf %in, %in_4 : f16
            linalg.yield %56 : f16
          }
          loom.semaphore_give %50 : memref<1x1024x32xf16>
          loom.semaphore_give %31 : memref<1x1024x128xf16>
          loom.sync ins(%24 : memref<1x1024x128xf16>) outs(%23 : memref<1x1024x128xf16>)
          loom.semaphore_give %24 : memref<1x1024x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<1x1024x128xf16>
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      }
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x2x4_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc2_dim_y_level0_bc8_dim_x_level0_bc2_dim_y_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8192 = arith.constant 8192 : index
      %c131072 = arith.constant 131072 : index
      %c524288 = arith.constant 524288 : index
      %c8 = arith.constant 8 : index
      %c7 = arith.constant 7 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c4 = arith.constant 4 : index
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c4, %c2, %c8) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c8 step %c1 {
          %20 = arith.muli %arg7, %c4 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = arith.muli %arg5, %c2 overflow<nsw> : index
          %23 = arith.addi %22, %arg6 : index
          %24 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %26 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %27 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %28 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %29 = arith.muli %21, %c524288 overflow<nsw> : index
          %30 = arith.muli %23, %c131072 : index
          %31 = arith.addi %29, %30 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          %32 = arith.muli %arg4, %c2 : index
          %33 = arith.addi %arg5, %32 : index
          loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
          loom.sync ins(%27 : memref<1x1024x128xf16>) outs(%28 : memref<1x1024x128xf16>)
          loom.semaphore_give %27 : memref<1x1024x128xf16>
          %34 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %35 = loom.semaphore_take %34 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          linalg.fill ins(%cst : f16) outs(%35 : memref<1x1024x128xf16>)
          %36 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%37 : memref<1x1024x1xf16>)
          %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x1024x1xf16>)
          %40 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %41 = loom.semaphore_take %40 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %44 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %45 = loom.semaphore_take %44 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %46 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %47 = loom.semaphore_take %46 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %48 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %49 = loom.semaphore_take %48 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %50 = loom.semaphore_take %48 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %51 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
          %52 = loom.semaphore_take %51 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
          %53 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
          %54 = loom.semaphore_take %53 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %55 = loom.semaphore_take %53 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %56 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %57 = loom.semaphore_take %56 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          %58 = loom.semaphore_take %56 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg8 = %c0 to %c64 step %c1 {
            %60 = arith.muli %arg8, %c64 : index
            %61 = arith.addi %29, %60 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%61], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
            %62 = arith.addi %32, %c1 : index
            loom.copy %reinterpret_cast_4, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 8] region : (UL : [%32, %c0], LR : [%62, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
            loom.sync ins(%50 : memref<1x128x64xf16>) outs(%49 : memref<1x128x64xf16>)
            loom.semaphore_give %50 : memref<1x128x64xf16>
            linalg.fill ins(%cst : f16) outs(%52 : memref<1x1024x64xf16>)
            linalg.batch_matmul ins(%28, %49 : memref<1x1024x128xf16>, memref<1x128x64xf16>) outs(%52 : memref<1x1024x64xf16>)
            loom.semaphore_give %49 : memref<1x128x64xf16>
            linalg.fill ins(%cst_1 : f16) outs(%43 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : memref<1x1024x64xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %66 = arith.maximumf %in, %out : f16
              linalg.yield %66 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %43 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.mulf %in_6, %cst_2 : f16
              %67 = arith.cmpf ogt, %in, %66 : f16
              %68 = arith.select %67, %in, %66 : f16
              linalg.yield %68 : f16
            }
            %63 = loom.broadcast ins(%43 : memref<1x1024x1xf16>) outs(%55 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52, %63 : memref<1x1024x64xf16>, memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>) outs(%52 : memref<1x1024x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.mulf %in, %cst_2 : f16
              %67 = arith.subf %66, %in_6 : f16
              %68 = math.exp %67 : f16
              linalg.yield %68 : f16
            }
            loom.semaphore_give %55 : memref<1x1024x32xf16>
            linalg.fill ins(%cst : f16) outs(%45 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : memref<1x1024x64xf16>) outs(%45 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %66 = arith.addf %in, %out : f16
              linalg.yield %66 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %43 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%47 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.subf %in, %in_6 : f16
              %67 = math.exp %66 : f16
              linalg.yield %67 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %47, %45 : memref<1x1024x1xf16>, memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%37 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %66 = arith.mulf %in, %in_6 : f16
              %67 = arith.addf %66, %in_7 : f16
              linalg.yield %67 : f16
            }
            loom.semaphore_give %45 : memref<1x1024x1xf16>
            %64 = arith.muli %arg8, %c8192 : index
            %65 = arith.addi %29, %64 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%65], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [2, 8] region : (UL : [%32, %c0], LR : [%62, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.sync ins(%58 : memref<1x64x128xf16>) outs(%57 : memref<1x64x128xf16>)
            loom.semaphore_give %58 : memref<1x64x128xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x1024x128xf16>)
            linalg.batch_matmul ins(%52, %57 : memref<1x1024x64xf16>, memref<1x64x128xf16>) outs(%41 : memref<1x1024x128xf16>)
            loom.semaphore_give %57 : memref<1x64x128xf16>
            loom.semaphore_give %52 : memref<1x1024x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %35, %47 : memref<1x1024x128xf16>, memref<1x1024x128xf16>, memref<1x1024x1xf16>) outs(%35 : memref<1x1024x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %66 = arith.mulf %in_6, %in_7 : f16
              %67 = arith.addf %in, %66 : f16
              linalg.yield %67 : f16
            }
            loom.semaphore_give %47 : memref<1x1024x1xf16>
            loom.semaphore_give %41 : memref<1x1024x128xf16>
            linalg.copy ins(%43 : memref<1x1024x1xf16>) outs(%39 : memref<1x1024x1xf16>)
            loom.semaphore_give %43 : memref<1x1024x1xf16>
          }
          loom.semaphore_give %39 : memref<1x1024x1xf16>
          loom.semaphore_give %28 : memref<1x1024x128xf16>
          %59 = loom.broadcast ins(%37 : memref<1x1024x1xf16>) outs(%54 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %37 : memref<1x1024x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %59 : memref<1x1024x128xf16>, memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%26 : memref<1x1024x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %60 = arith.divf %in, %in_4 : f16
            linalg.yield %60 : f16
          }
          loom.semaphore_give %54 : memref<1x1024x32xf16>
          loom.semaphore_give %35 : memref<1x1024x128xf16>
          loom.sync ins(%26 : memref<1x1024x128xf16>) outs(%25 : memref<1x1024x128xf16>)
          loom.semaphore_give %26 : memref<1x1024x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %25, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %25 : memref<1x1024x128xf16>
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x4x2_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level0_bc4_dim_y_level0_bc8_dim_x_level0_bc4_dim_y_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8192 = arith.constant 8192 : index
      %c131072 = arith.constant 131072 : index
      %c524288 = arith.constant 524288 : index
      %c16 = arith.constant 16 : index
      %c7 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5, %arg6) = (%c0, %c0, %c0) to (%c2, %c4, %c8) step (%c1, %c1, %c1) {
        scf.for %arg7 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg7, %c2 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = arith.muli %arg5, %c4 overflow<nsw> : index
          %23 = arith.addi %22, %arg6 : index
          %24 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %25 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %26 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %27 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %28 = loom.semaphore_take %24 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %29 = arith.muli %21, %c524288 overflow<nsw> : index
          %30 = arith.muli %23, %c131072 : index
          %31 = arith.addi %29, %30 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          %32 = arith.muli %arg4, %c4 : index
          %33 = arith.addi %arg5, %32 : index
          loom.copy %reinterpret_cast, %27 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
          loom.sync ins(%27 : memref<1x1024x128xf16>) outs(%28 : memref<1x1024x128xf16>)
          loom.semaphore_give %27 : memref<1x1024x128xf16>
          %34 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %35 = loom.semaphore_take %34 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          linalg.fill ins(%cst : f16) outs(%35 : memref<1x1024x128xf16>)
          %36 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %37 = loom.semaphore_take %36 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%37 : memref<1x1024x1xf16>)
          %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x1024x1xf16>)
          %40 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %41 = loom.semaphore_take %40 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %44 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %45 = loom.semaphore_take %44 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %46 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %47 = loom.semaphore_take %46 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %48 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %49 = loom.semaphore_take %48 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %50 = loom.semaphore_take %48 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %51 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
          %52 = loom.semaphore_take %51 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
          %53 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
          %54 = loom.semaphore_take %53 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %55 = loom.semaphore_take %53 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %56 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %57 = loom.semaphore_take %56 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          %58 = loom.semaphore_take %56 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg8 = %c0 to %c64 step %c1 {
            %60 = arith.muli %arg8, %c64 : index
            %61 = arith.addi %29, %60 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%61], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
            %62 = arith.addi %32, %c3 : index
            loom.copy %reinterpret_cast_4, %50 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%32, %c0], LR : [%62, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
            loom.sync ins(%50 : memref<1x128x64xf16>) outs(%49 : memref<1x128x64xf16>)
            loom.semaphore_give %50 : memref<1x128x64xf16>
            linalg.fill ins(%cst : f16) outs(%52 : memref<1x1024x64xf16>)
            linalg.batch_matmul ins(%28, %49 : memref<1x1024x128xf16>, memref<1x128x64xf16>) outs(%52 : memref<1x1024x64xf16>)
            loom.semaphore_give %49 : memref<1x128x64xf16>
            linalg.fill ins(%cst_1 : f16) outs(%43 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : memref<1x1024x64xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %66 = arith.maximumf %in, %out : f16
              linalg.yield %66 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %43 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.mulf %in_6, %cst_2 : f16
              %67 = arith.cmpf ogt, %in, %66 : f16
              %68 = arith.select %67, %in, %66 : f16
              linalg.yield %68 : f16
            }
            %63 = loom.broadcast ins(%43 : memref<1x1024x1xf16>) outs(%55 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%52, %63 : memref<1x1024x64xf16>, memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>) outs(%52 : memref<1x1024x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.mulf %in, %cst_2 : f16
              %67 = arith.subf %66, %in_6 : f16
              %68 = math.exp %67 : f16
              linalg.yield %68 : f16
            }
            loom.semaphore_give %55 : memref<1x1024x32xf16>
            linalg.fill ins(%cst : f16) outs(%45 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%52 : memref<1x1024x64xf16>) outs(%45 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %66 = arith.addf %in, %out : f16
              linalg.yield %66 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%39, %43 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%47 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %66 = arith.subf %in, %in_6 : f16
              %67 = math.exp %66 : f16
              linalg.yield %67 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %47, %45 : memref<1x1024x1xf16>, memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%37 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %66 = arith.mulf %in, %in_6 : f16
              %67 = arith.addf %66, %in_7 : f16
              linalg.yield %67 : f16
            }
            loom.semaphore_give %45 : memref<1x1024x1xf16>
            %64 = arith.muli %arg8, %c8192 : index
            %65 = arith.addi %29, %64 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%65], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %58 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [4, 8] region : (UL : [%32, %c0], LR : [%62, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.sync ins(%58 : memref<1x64x128xf16>) outs(%57 : memref<1x64x128xf16>)
            loom.semaphore_give %58 : memref<1x64x128xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x1024x128xf16>)
            linalg.batch_matmul ins(%52, %57 : memref<1x1024x64xf16>, memref<1x64x128xf16>) outs(%41 : memref<1x1024x128xf16>)
            loom.semaphore_give %57 : memref<1x64x128xf16>
            loom.semaphore_give %52 : memref<1x1024x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%41, %35, %47 : memref<1x1024x128xf16>, memref<1x1024x128xf16>, memref<1x1024x1xf16>) outs(%35 : memref<1x1024x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %66 = arith.mulf %in_6, %in_7 : f16
              %67 = arith.addf %in, %66 : f16
              linalg.yield %67 : f16
            }
            loom.semaphore_give %47 : memref<1x1024x1xf16>
            loom.semaphore_give %41 : memref<1x1024x128xf16>
            linalg.copy ins(%43 : memref<1x1024x1xf16>) outs(%39 : memref<1x1024x1xf16>)
            loom.semaphore_give %43 : memref<1x1024x1xf16>
          }
          loom.semaphore_give %39 : memref<1x1024x1xf16>
          loom.semaphore_give %28 : memref<1x1024x128xf16>
          %59 = loom.broadcast ins(%37 : memref<1x1024x1xf16>) outs(%54 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %37 : memref<1x1024x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %59 : memref<1x1024x128xf16>, memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%26 : memref<1x1024x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %60 = arith.divf %in, %in_4 : f16
            linalg.yield %60 : f16
          }
          loom.semaphore_give %54 : memref<1x1024x32xf16>
          loom.semaphore_give %35 : memref<1x1024x128xf16>
          loom.sync ins(%26 : memref<1x1024x128xf16>) outs(%25 : memref<1x1024x128xf16>)
          loom.semaphore_give %26 : memref<1x1024x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%31], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %25, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%33, %arg6], LR : [%33, %arg6]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %25 : memref<1x1024x128xf16>
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [1, 0, 0], loom.physical_dims = [@dim_x, @dim_x, @dim_y]}
      return
    }
  }
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 32 : index}, loom.tile_m = {is_reduction = false, upper_bound = 4096 : index}, loom.tile_n = {is_reduction = false, upper_bound = 4096 : index}} {
    func.func @attention__x8x1_y8__d0i1_d1i1_d2i0__f01__n_dim_x_level1_bc8_dim_y_level0_bc8_dim_x_level1_bc8_dim_y_level0_bc8_n__tile_b1__tile_m1024__tile_n64(%arg0: memref<32x128x4096xf16>, %arg1: memref<32x4096x128xf16>, %arg2: memref<32x4096x128xf16>, %arg3: memref<32x4096x128xf16>) {
      %c8192 = arith.constant 8192 : index
      %c131072 = arith.constant 131072 : index
      %c524288 = arith.constant 524288 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
      %c32 = arith.constant 32 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %20 = arith.muli %arg4, %c8 overflow<nsw> : index
          %21 = arith.addi %20, %arg5 : index
          %22 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %24 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %25 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %26 = loom.semaphore_take %22 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %27 = arith.muli %arg6, %c524288 overflow<nsw> : index
          %28 = arith.muli %21, %c131072 : index
          %29 = arith.addi %27, %28 : index
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%29], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x1024x128xf16>
          loom.sync ins(%25 : memref<1x1024x128xf16>) outs(%26 : memref<1x1024x128xf16>)
          loom.semaphore_give %25 : memref<1x1024x128xf16>
          %30 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %31 = loom.semaphore_take %30 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          linalg.fill ins(%cst : f16) outs(%31 : memref<1x1024x128xf16>)
          %32 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %33 = loom.semaphore_take %32 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%33 : memref<1x1024x1xf16>)
          %34 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %35 = loom.semaphore_take %34 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%35 : memref<1x1024x1xf16>)
          %36 = loom.alloc [1, 1024, 128] on @L1 : memref<1x1024x128xf16>
          %37 = loom.semaphore_take %36 : memref<1x1024x128xf16> -> memref<1x1024x128xf16>
          %38 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %39 = loom.semaphore_take %38 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %40 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %41 = loom.semaphore_take %40 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %42 = loom.alloc [1, 1024, 1] on @L1 : memref<1x1024x1xf16>
          %43 = loom.semaphore_take %42 : memref<1x1024x1xf16> -> memref<1x1024x1xf16>
          %44 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %45 = loom.semaphore_take %44 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %46 = loom.semaphore_take %44 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %47 = loom.alloc [1, 1024, 64] on @L1 : memref<1x1024x64xf16>
          %48 = loom.semaphore_take %47 : memref<1x1024x64xf16> -> memref<1x1024x64xf16>
          %49 = loom.alloc [1, 1024, 32] on @L1 : memref<1x1024x32xf16>
          %50 = loom.semaphore_take %49 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %51 = loom.semaphore_take %49 : memref<1x1024x32xf16> -> memref<1x1024x32xf16>
          %52 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %53 = loom.semaphore_take %52 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          %54 = loom.semaphore_take %52 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg7 = %c0 to %c64 step %c1 {
            %56 = arith.muli %arg7, %c64 : index
            %57 = arith.addi %27, %56 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 64], strides: [524288, 4096, 1] : memref<32x128x4096xf16> to memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x128x64xf16, strided<[524288, 4096, 1], offset: ?>> to memref<1x128x64xf16>
            loom.sync ins(%46 : memref<1x128x64xf16>) outs(%45 : memref<1x128x64xf16>)
            loom.semaphore_give %46 : memref<1x128x64xf16>
            linalg.fill ins(%cst : f16) outs(%48 : memref<1x1024x64xf16>)
            linalg.batch_matmul ins(%26, %45 : memref<1x1024x128xf16>, memref<1x128x64xf16>) outs(%48 : memref<1x1024x64xf16>)
            loom.semaphore_give %45 : memref<1x128x64xf16>
            linalg.fill ins(%cst_1 : f16) outs(%39 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x1024x64xf16>) outs(%39 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.maximumf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%39 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in_6, %cst_2 : f16
              %62 = arith.cmpf ogt, %in, %61 : f16
              %63 = arith.select %62, %in, %61 : f16
              linalg.yield %63 : f16
            }
            %58 = loom.broadcast ins(%39 : memref<1x1024x1xf16>) outs(%51 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %58 : memref<1x1024x64xf16>, memref<1x1024x64xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<1x1024x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.mulf %in, %cst_2 : f16
              %62 = arith.subf %61, %in_6 : f16
              %63 = math.exp %62 : f16
              linalg.yield %63 : f16
            }
            loom.semaphore_give %51 : memref<1x1024x32xf16>
            linalg.fill ins(%cst : f16) outs(%41 : memref<1x1024x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x1024x64xf16>) outs(%41 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %61 = arith.addf %in, %out : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%43 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %61 = arith.subf %in, %in_6 : f16
              %62 = math.exp %61 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %43, %41 : memref<1x1024x1xf16>, memref<1x1024x1xf16>, memref<1x1024x1xf16>) outs(%33 : memref<1x1024x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in, %in_6 : f16
              %62 = arith.addf %61, %in_7 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %41 : memref<1x1024x1xf16>
            %59 = arith.muli %arg7, %c8192 : index
            %60 = arith.addi %27, %59 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%60], sizes: [1, 64, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 8] region : (UL : [%c0, %c0], LR : [%c7, %c7]) : memref<1x64x128xf16, strided<[524288, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.sync ins(%54 : memref<1x64x128xf16>) outs(%53 : memref<1x64x128xf16>)
            loom.semaphore_give %54 : memref<1x64x128xf16>
            linalg.fill ins(%cst : f16) outs(%37 : memref<1x1024x128xf16>)
            linalg.batch_matmul ins(%48, %53 : memref<1x1024x64xf16>, memref<1x64x128xf16>) outs(%37 : memref<1x1024x128xf16>)
            loom.semaphore_give %53 : memref<1x64x128xf16>
            loom.semaphore_give %48 : memref<1x1024x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %31, %43 : memref<1x1024x128xf16>, memref<1x1024x128xf16>, memref<1x1024x1xf16>) outs(%31 : memref<1x1024x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %61 = arith.mulf %in_6, %in_7 : f16
              %62 = arith.addf %in, %61 : f16
              linalg.yield %62 : f16
            }
            loom.semaphore_give %43 : memref<1x1024x1xf16>
            loom.semaphore_give %37 : memref<1x1024x128xf16>
            linalg.copy ins(%39 : memref<1x1024x1xf16>) outs(%35 : memref<1x1024x1xf16>)
            loom.semaphore_give %39 : memref<1x1024x1xf16>
          }
          loom.semaphore_give %35 : memref<1x1024x1xf16>
          loom.semaphore_give %26 : memref<1x1024x128xf16>
          %55 = loom.broadcast ins(%33 : memref<1x1024x1xf16>) outs(%50 : memref<1x1024x32xf16>) dim(2) -> memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %33 : memref<1x1024x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %55 : memref<1x1024x128xf16>, memref<1x1024x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%24 : memref<1x1024x128xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %56 = arith.divf %in, %in_4 : f16
            linalg.yield %56 : f16
          }
          loom.semaphore_give %50 : memref<1x1024x32xf16>
          loom.semaphore_give %31 : memref<1x1024x128xf16>
          loom.sync ins(%24 : memref<1x1024x128xf16>) outs(%23 : memref<1x1024x128xf16>)
          loom.semaphore_give %24 : memref<1x1024x128xf16>
          %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%29], sizes: [1, 1024, 128], strides: [524288, 128, 1] : memref<32x4096x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.copy %23, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg4, %arg5], LR : [%arg4, %arg5]) : memref<1x1024x128xf16> to memref<1x1024x128xf16, strided<[524288, 128, 1], offset: ?>>
          loom.semaphore_give %23 : memref<1x1024x128xf16>
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      }
      return
    }
  }
}
