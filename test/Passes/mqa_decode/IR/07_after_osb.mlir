module attributes {loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
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
  module attributes {loom.pass_name = "Materialize", loom.tile_b = {is_reduction = false, upper_bound = 16 : index}, loom.tile_n = {is_reduction = false, upper_bound = 64 : index}, loom.tile_s = {is_reduction = false, upper_bound = 8192 : index}} {
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_dim_x_level0_bc8__tile_b2__tile_n64__tile_s1024(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c2097152 = arith.constant 2097152 : index
      %c8192 = arith.constant 8192 : index
      %c128 = arith.constant 128 : index
      %c16 = arith.constant 16 : index
      %c7 = arith.constant 7 : index
      %cst = arith.constant 2.000000e+00 : f16
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst_0 = arith.constant 0.000000e+00 : f16
      %cst_1 = arith.constant 1.000000e+00 : f16
      %cst_2 = arith.constant 0xFC00 : f16
      %cst_3 = arith.constant 1.275630e-01 : f16
      %c1024 = arith.constant 1024 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      scf.parallel (%arg4, %arg5) = (%c0, %c0) to (%c8, %c8) step (%c1, %c1) {
        %20 = loom.alloc [2, 32, 128] on @L1 : memref<2x32x128xf16>
        %21 = loom.semaphore_take %20 : memref<2x32x128xf16> -> memref<2x32x128xf16>
        %22 = loom.semaphore_take %20 : memref<2x32x128xf16> -> memref<2x32x128xf16>
        %23 = arith.muli %arg4, %c8192 : index
        %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%23], sizes: [2, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>>
        loom.copy %reinterpret_cast, %22 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<2x32x128xf16>
        %24 = arith.muli %arg5, %c1024 : index
        %25 = loom.alloc [2, 32, 128] on @L1 : memref<2x32x128xf16>
        %26 = loom.semaphore_take %25 : memref<2x32x128xf16> -> memref<2x32x128xf16>
        linalg.fill ins(%cst_0 : f16) outs(%26 : memref<2x32x128xf16>)
        %27 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
        %28 = loom.semaphore_take %27 : memref<2x32xf16> -> memref<2x32xf16>
        %29 = loom.semaphore_take %27 : memref<2x32xf16> -> memref<2x32xf16>
        %30 = loom.semaphore_take %27 : memref<2x32xf16> -> memref<2x32xf16>
        %31 = loom.semaphore_take %27 : memref<2x32xf16> -> memref<2x32xf16>
        linalg.fill ins(%cst_1 : f16) outs(%31 : memref<2x32xf16>)
        %32 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
        %33 = loom.semaphore_take %32 : memref<2x32xf16> -> memref<2x32xf16>
        linalg.fill ins(%cst_2 : f16) outs(%33 : memref<2x32xf16>)
        %34 = loom.alloc [2, 32, 128] on @L1 : memref<2x32x128xf16>
        %35 = loom.semaphore_take %34 : memref<2x32x128xf16> -> memref<2x32x128xf16>
        %36 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
        %37 = loom.semaphore_take %36 : memref<2x32xf16> -> memref<2x32xf16>
        %38 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
        %39 = loom.semaphore_take %38 : memref<2x32xf16> -> memref<2x32xf16>
        %40 = loom.alloc [2, 32] on @L1 : memref<2x32xf16>
        %41 = loom.semaphore_take %40 : memref<2x32xf16> -> memref<2x32xf16>
        %42 = loom.alloc [2, 128, 64] on @L1 : memref<2x128x64xf16>
        %43 = loom.semaphore_take %42 : memref<2x128x64xf16> -> memref<2x128x64xf16>
        %44 = loom.alloc [2, 32, 64] on @L1 : memref<2x32x64xf16>
        %45 = loom.semaphore_take %44 : memref<2x32x64xf16> -> memref<2x32x64xf16>
        %46 = loom.alloc [2, 64, 128] on @L1 : memref<2x64x128xf16>
        %47 = loom.semaphore_take %46 : memref<2x64x128xf16> -> memref<2x64x128xf16>
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %53 = arith.muli %arg6, %c64 : index
          %54 = arith.addi %24, %53 : index
          %55 = arith.muli %arg4, %c2097152 : index
          %56 = arith.addi %55, %54 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%56], sizes: [2, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<2x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %43 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<2x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<2x128x64xf16>
          linalg.fill ins(%cst_0 : f16) outs(%45 : memref<2x32x64xf16>)
          linalg.batch_matmul ins(%22, %43 : memref<2x32x128xf16>, memref<2x128x64xf16>) outs(%45 : memref<2x32x64xf16>)
          loom.semaphore_give %43 : memref<2x128x64xf16>
          linalg.fill ins(%cst_2 : f16) outs(%37 : memref<2x32xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<2x32x64xf16>) outs(%37 : memref<2x32xf16>) {
          ^bb0(%in: f16, %out: f16):
            %59 = arith.maximumf %in, %out : f16
            linalg.yield %59 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<2x32xf16>, memref<2x32xf16>) outs(%37 : memref<2x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %59 = arith.mulf %in_6, %cst_3 : f16
            %60 = arith.cmpf ogt, %in, %59 : f16
            %61 = arith.select %60, %in, %59 : f16
            linalg.yield %61 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%45, %37 : memref<2x32x64xf16>, memref<2x32xf16>) outs(%45 : memref<2x32x64xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %59 = arith.mulf %in, %cst_3 : f16
            %60 = arith.subf %59, %in_6 : f16
            %61 = math.powf %cst, %60 : f16
            linalg.yield %61 : f16
          }
          linalg.fill ins(%cst_0 : f16) outs(%39 : memref<2x32xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%45 : memref<2x32x64xf16>) outs(%39 : memref<2x32xf16>) {
          ^bb0(%in: f16, %out: f16):
            %59 = arith.addf %in, %out : f16
            linalg.yield %59 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%33, %37 : memref<2x32xf16>, memref<2x32xf16>) outs(%41 : memref<2x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %out: f16):
            %59 = arith.subf %in, %in_6 : f16
            %60 = math.powf %cst, %59 : f16
            linalg.yield %60 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %41, %39 : memref<2x32xf16>, memref<2x32xf16>, memref<2x32xf16>) outs(%31 : memref<2x32xf16>) {
          ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
            %59 = arith.mulf %in, %in_6 : f16
            %60 = arith.addf %59, %in_7 : f16
            linalg.yield %60 : f16
          }
          loom.semaphore_give %39 : memref<2x32xf16>
          %57 = arith.muli %54, %c128 overflow<nsw> : index
          %58 = arith.addi %55, %57 : index
          %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%58], sizes: [2, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<2x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_5, %47 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<2x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<2x64x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%35 : memref<2x32x128xf16>)
          linalg.batch_matmul ins(%45, %47 : memref<2x32x64xf16>, memref<2x64x128xf16>) outs(%35 : memref<2x32x128xf16>)
          loom.semaphore_give %47 : memref<2x64x128xf16>
          loom.semaphore_give %45 : memref<2x32x64xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %26, %41 : memref<2x32x128xf16>, memref<2x32x128xf16>, memref<2x32xf16>) outs(%26 : memref<2x32x128xf16>) {
          ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
            %59 = arith.mulf %in_6, %in_7 : f16
            %60 = arith.addf %in, %59 : f16
            linalg.yield %60 : f16
          }
          loom.semaphore_give %41 : memref<2x32xf16>
          loom.semaphore_give %35 : memref<2x32x128xf16>
          linalg.copy ins(%37 : memref<2x32xf16>) outs(%33 : memref<2x32xf16>)
          loom.semaphore_give %37 : memref<2x32xf16>
        } {loom.iter_type = #loom.iter_type<sequential>}
        loom.semaphore_give %22 : memref<2x32x128xf16>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%31, %33 : memref<2x32xf16>, memref<2x32xf16>) outs(%30 : memref<2x32xf16>) {
        ^bb0(%in: f16, %in_4: f16, %out: f16):
          %53 = math.log2 %in : f16
          %54 = arith.addf %53, %in_4 : f16
          linalg.yield %54 : f16
        }
        loom.semaphore_give %31 : memref<2x32xf16>
        loom.semaphore_give %33 : memref<2x32xf16>
        %48 = loom.alloc [8, 2, 32] on @L1 : memref<8x2x32xf16>
        %49 = loom.semaphore_take %48 : memref<8x2x32xf16> -> memref<8x2x32xf16>
        loom.gather ins(%30 : memref<2x32xf16>) outs(%49 : memref<8x2x32xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
        loom.semaphore_give %30 : memref<2x32xf16>
        %50 = loom.alloc [8, 2, 32, 128] on @L1 : memref<8x2x32x128xf16>
        %51 = loom.semaphore_take %50 : memref<8x2x32x128xf16> -> memref<8x2x32x128xf16>
        loom.gather ins(%26 : memref<2x32x128xf16>) outs(%51 : memref<8x2x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
        loom.semaphore_give %26 : memref<2x32x128xf16>
        %52 = arith.cmpi eq, %arg5, %c0 : index
        scf.if %52 {
          linalg.fill ins(%cst_2 : f16) outs(%29 : memref<2x32xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%49 : memref<8x2x32xf16>) outs(%29 : memref<2x32xf16>) {
          ^bb0(%in: f16, %out: f16):
            %53 = arith.maximumf %in, %out : f16
            linalg.yield %53 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %29 : memref<8x2x32xf16>, memref<2x32xf16>) outs(%49 : memref<8x2x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %53 = arith.subf %in, %in_5 : f16
            %54 = math.powf %cst, %53 : f16
            linalg.yield %54 : f16
          }
          loom.semaphore_give %29 : memref<2x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%28 : memref<2x32xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%49 : memref<8x2x32xf16>) outs(%28 : memref<2x32xf16>) {
          ^bb0(%in: f16, %out: f16):
            %53 = arith.addf %in, %out : f16
            linalg.yield %53 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%49, %28 : memref<8x2x32xf16>, memref<2x32xf16>) outs(%49 : memref<8x2x32xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %53 = arith.divf %in, %in_5 : f16
            linalg.yield %53 : f16
          }
          loom.semaphore_give %28 : memref<2x32xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%51, %49 : memref<8x2x32x128xf16>, memref<8x2x32xf16>) outs(%51 : memref<8x2x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %53 = arith.mulf %in, %in_5 : f16
            linalg.yield %53 : f16
          }
          loom.semaphore_give %49 : memref<8x2x32xf16>
          linalg.fill ins(%cst_0 : f16) outs(%21 : memref<2x32x128xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%51 : memref<8x2x32x128xf16>) outs(%21 : memref<2x32x128xf16>) {
          ^bb0(%in: f16, %out: f16):
            %53 = arith.addf %in, %out : f16
            linalg.yield %53 : f16
          }
          loom.semaphore_give %51 : memref<8x2x32x128xf16>
          %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [2, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.copy %21, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<2x32x128xf16> to memref<2x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.semaphore_give %21 : memref<2x32x128xf16>
        }
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
}
