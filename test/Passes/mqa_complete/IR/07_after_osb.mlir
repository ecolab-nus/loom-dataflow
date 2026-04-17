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
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_dim_x_level0_bc8__tile_b1__tile_n64__tile_s1024(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
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
        scf.for %arg6 = %c0 to %c2 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg4, %20 : index
          %22 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %25 = arith.muli %21, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %26 = arith.muli %arg5, %c1024 : index
          %27 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %28 = loom.semaphore_take %27 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst_0 : f16) outs(%28 : memref<1x32x128xf16>)
          %29 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
          %30 = loom.semaphore_take %29 : memref<1x32xf16> -> memref<1x32xf16>
          %31 = loom.semaphore_take %29 : memref<1x32xf16> -> memref<1x32xf16>
          %32 = loom.semaphore_take %29 : memref<1x32xf16> -> memref<1x32xf16>
          linalg.fill ins(%cst_1 : f16) outs(%32 : memref<1x32xf16>)
          %33 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
          %34 = loom.semaphore_take %33 : memref<1x32xf16> -> memref<1x32xf16>
          linalg.fill ins(%cst_2 : f16) outs(%34 : memref<1x32xf16>)
          %35 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %36 = loom.semaphore_take %35 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %37 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
          %38 = loom.semaphore_take %37 : memref<1x32xf16> -> memref<1x32xf16>
          %39 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
          %40 = loom.semaphore_take %39 : memref<1x32xf16> -> memref<1x32xf16>
          %41 = loom.alloc [1, 32] on @L1 : memref<1x32xf16>
          %42 = loom.semaphore_take %41 : memref<1x32xf16> -> memref<1x32xf16>
          %43 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %44 = loom.semaphore_take %43 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %45 = loom.alloc [1, 32, 64] on @L1 : memref<1x32x64xf16>
          %46 = loom.semaphore_take %45 : memref<1x32x64xf16> -> memref<1x32x64xf16>
          %47 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %48 = loom.semaphore_take %47 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %54 = arith.muli %arg7, %c64 : index
            %55 = arith.addi %26, %54 : index
            %56 = arith.muli %21, %c1048576 overflow<nsw> : index
            %57 = arith.addi %56, %55 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%57], sizes: [1, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %44 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x64xf16>
            linalg.fill ins(%cst_0 : f16) outs(%46 : memref<1x32x64xf16>)
            linalg.batch_matmul ins(%24, %44 : memref<1x32x128xf16>, memref<1x128x64xf16>) outs(%46 : memref<1x32x64xf16>)
            loom.semaphore_give %44 : memref<1x128x64xf16>
            linalg.fill ins(%cst_2 : f16) outs(%38 : memref<1x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x64xf16>) outs(%38 : memref<1x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %60 = arith.maximumf %in, %out : f16
              linalg.yield %60 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%34, %38 : memref<1x32xf16>, memref<1x32xf16>) outs(%38 : memref<1x32xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %60 = arith.mulf %in_6, %cst_3 : f16
              %61 = arith.cmpf ogt, %in, %60 : f16
              %62 = arith.select %61, %in, %60 : f16
              linalg.yield %62 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%46, %38 : memref<1x32x64xf16>, memref<1x32xf16>) outs(%46 : memref<1x32x64xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %60 = arith.mulf %in, %cst_3 : f16
              %61 = arith.subf %60, %in_6 : f16
              %62 = math.powf %cst, %61 : f16
              linalg.yield %62 : f16
            }
            linalg.fill ins(%cst_0 : f16) outs(%40 : memref<1x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%46 : memref<1x32x64xf16>) outs(%40 : memref<1x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %60 = arith.addf %in, %out : f16
              linalg.yield %60 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%34, %38 : memref<1x32xf16>, memref<1x32xf16>) outs(%42 : memref<1x32xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %60 = arith.subf %in, %in_6 : f16
              %61 = math.powf %cst, %60 : f16
              linalg.yield %61 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32, %42, %40 : memref<1x32xf16>, memref<1x32xf16>, memref<1x32xf16>) outs(%32 : memref<1x32xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %60 = arith.mulf %in, %in_6 : f16
              %61 = arith.addf %60, %in_7 : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %40 : memref<1x32xf16>
            %58 = arith.muli %55, %c128 overflow<nsw> : index
            %59 = arith.addi %56, %58 : index
            %reinterpret_cast_5 = memref.reinterpret_cast %arg1 to offset: [%59], sizes: [1, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_5, %48 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x64x128xf16>
            linalg.fill ins(%cst_0 : f16) outs(%36 : memref<1x32x128xf16>)
            linalg.batch_matmul ins(%46, %48 : memref<1x32x64xf16>, memref<1x64x128xf16>) outs(%36 : memref<1x32x128xf16>)
            loom.semaphore_give %48 : memref<1x64x128xf16>
            loom.semaphore_give %46 : memref<1x32x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %28, %42 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32xf16>) outs(%28 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %60 = arith.mulf %in_6, %in_7 : f16
              %61 = arith.addf %in, %60 : f16
              linalg.yield %61 : f16
            }
            loom.semaphore_give %42 : memref<1x32xf16>
            loom.semaphore_give %36 : memref<1x32x128xf16>
            linalg.copy ins(%38 : memref<1x32xf16>) outs(%34 : memref<1x32xf16>)
            loom.semaphore_give %38 : memref<1x32xf16>
          } {loom.iter_type = #loom.iter_type<sequential>}
          loom.semaphore_give %24 : memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%32, %34 : memref<1x32xf16>, memref<1x32xf16>) outs(%31 : memref<1x32xf16>) {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %54 = math.log2 %in : f16
            %55 = arith.addf %54, %in_4 : f16
            linalg.yield %55 : f16
          }
          loom.semaphore_give %32 : memref<1x32xf16>
          loom.semaphore_give %34 : memref<1x32xf16>
          %49 = loom.alloc [8, 1, 32] on @L1 : memref<8x1x32xf16>
          %50 = loom.semaphore_take %49 : memref<8x1x32xf16> -> memref<8x1x32xf16>
          loom.gather ins(%31 : memref<1x32xf16>) outs(%50 : memref<8x1x32xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
          loom.semaphore_give %31 : memref<1x32xf16>
          %51 = loom.alloc [8, 1, 32, 128] on @L1 : memref<8x1x32x128xf16>
          %52 = loom.semaphore_take %51 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
          loom.gather ins(%28 : memref<1x32x128xf16>) outs(%52 : memref<8x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
          loom.semaphore_give %28 : memref<1x32x128xf16>
          %53 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %53 {
            linalg.fill ins(%cst_2 : f16) outs(%30 : memref<1x32xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>], iterator_types = ["reduction", "parallel", "parallel"]} ins(%50 : memref<8x1x32xf16>) outs(%30 : memref<1x32xf16>) {
            ^bb0(%in: f16, %out: f16):
              %54 = arith.maximumf %in, %out : f16
              linalg.yield %54 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%50, %30 : memref<8x1x32xf16>, memref<1x32xf16>) outs(%50 : memref<8x1x32xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %54 = arith.subf %in, %in_5 : f16
              %55 = math.powf %cst, %54 : f16
              linalg.yield %55 : f16
            }
            loom.semaphore_give %30 : memref<1x32xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%52, %50 : memref<8x1x32x128xf16>, memref<8x1x32xf16>) outs(%52 : memref<8x1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %54 = arith.mulf %in, %in_5 : f16
              linalg.yield %54 : f16
            }
            loom.semaphore_give %50 : memref<8x1x32xf16>
            linalg.fill ins(%cst_0 : f16) outs(%23 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%52 : memref<8x1x32x128xf16>) outs(%23 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %54 = arith.addf %in, %out : f16
              linalg.yield %54 : f16
            }
            loom.semaphore_give %52 : memref<8x1x32x128xf16>
            %reinterpret_cast_4 = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.copy %23, %reinterpret_cast_4 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %23 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
}
