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
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_n__tile_b1__tile_n64__tile_s1024(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c1048576 = arith.constant 1048576 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c16 = arith.constant 16 : index
      %c2 = arith.constant 2 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f16
      %cst_0 = arith.constant 1.000000e+00 : f16
      %cst_1 = arith.constant 0xFC00 : f16
      %c1 = arith.constant 1 : index
      %cst_2 = arith.constant 8.837890e-02 : f16
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
          %25 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %26 = arith.muli %21, %c4096 overflow<nsw> : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %25 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<1x32x128xf16>
          %27 = arith.muli %arg5, %c1024 : index
          %28 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %29 = loom.semaphore_take %28 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%29 : memref<1x32x128xf16>)
          %30 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %31 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %32 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %33 = loom.semaphore_take %30 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%33 : memref<1x32x1xf16>)
          %34 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %35 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %36 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%36 : memref<1x32x1xf16>)
          %37 = loom.alloc [1, 32, 128] on @L1 : memref<1x32x128xf16>
          %38 = loom.semaphore_take %37 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          %39 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %40 = loom.semaphore_take %39 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %41 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %42 = loom.semaphore_take %41 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %43 = loom.alloc [1, 32, 1] on @L1 : memref<1x32x1xf16>
          %44 = loom.semaphore_take %43 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          %45 = loom.alloc [1, 128, 64] on @L1 : memref<1x128x64xf16>
          %46 = loom.semaphore_take %45 : memref<1x128x64xf16> -> memref<1x128x64xf16>
          %47 = loom.alloc [1, 32, 64] on @L1 : memref<1x32x64xf16>
          %48 = loom.semaphore_take %47 : memref<1x32x64xf16> -> memref<1x32x64xf16>
          %49 = loom.alloc [1, 32, 32] on @L1 : memref<1x32x32xf16>
          %50 = loom.semaphore_take %49 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %51 = loom.semaphore_take %49 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %52 = loom.semaphore_take %49 : memref<1x32x32xf16> -> memref<1x32x32xf16>
          %53 = loom.alloc [1, 64, 128] on @L1 : memref<1x64x128xf16>
          %54 = loom.semaphore_take %53 : memref<1x64x128xf16> -> memref<1x64x128xf16>
          scf.for %arg7 = %c0 to %c16 step %c1 {
            %65 = arith.muli %arg7, %c64 : index
            %66 = arith.addi %27, %65 : index
            %67 = arith.muli %21, %c1048576 overflow<nsw> : index
            %68 = arith.addi %67, %66 : index
            %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%68], sizes: [1, 128, 64], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>>
            loom.copy %reinterpret_cast_3, %46 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x128x64xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<1x128x64xf16>
            loom.batch_matmul ins(%25, %46 : memref<1x32x128xf16>, memref<1x128x64xf16>) outs(%48 : memref<1x32x64xf16>)
            loom.semaphore_give %46 : memref<1x128x64xf16>
            linalg.fill ins(%cst_1 : f16) outs(%40 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x64xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %73 = arith.maximumf %in, %out : f16
              linalg.yield %73 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%40 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %73 = arith.mulf %in_5, %cst_2 : f16
              %74 = arith.cmpf ogt, %in, %73 : f16
              %75 = arith.select %74, %in, %73 : f16
              linalg.yield %75 : f16
            }
            %69 = loom.broadcast ins(%40 : memref<1x32x1xf16>) outs(%52 : memref<1x32x32xf16>) dim(2) -> memref<1x32x64xf16, strided<[?, ?, ?], offset: ?>>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48, %69 : memref<1x32x64xf16>, memref<1x32x64xf16, strided<[?, ?, ?], offset: ?>>) outs(%48 : memref<1x32x64xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %73 = arith.mulf %in, %cst_2 : f16
              %74 = arith.subf %73, %in_5 : f16
              %75 = math.exp %74 : f16
              linalg.yield %75 : f16
            }
            loom.semaphore_give %52 : memref<1x32x32xf16>
            linalg.fill ins(%cst : f16) outs(%42 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%48 : memref<1x32x64xf16>) outs(%42 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %73 = arith.addf %in, %out : f16
              linalg.yield %73 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%36, %40 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%44 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %out: f16):
              %73 = arith.subf %in, %in_5 : f16
              %74 = math.exp %73 : f16
              linalg.yield %74 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %44, %42 : memref<1x32x1xf16>, memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%33 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %73 = arith.mulf %in, %in_5 : f16
              %74 = arith.addf %73, %in_6 : f16
              linalg.yield %74 : f16
            }
            loom.semaphore_give %42 : memref<1x32x1xf16>
            %70 = loom.broadcast ins(%44 : memref<1x32x1xf16>) outs(%51 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
            loom.semaphore_give %44 : memref<1x32x1xf16>
            %71 = arith.muli %66, %c128 overflow<nsw> : index
            %72 = arith.addi %67, %71 : index
            %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%72], sizes: [1, 64, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>>
            loom.copy %reinterpret_cast_4, %54 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x64x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<1x64x128xf16>
            loom.batch_matmul ins(%48, %54 : memref<1x32x64xf16>, memref<1x64x128xf16>) outs(%38 : memref<1x32x128xf16>)
            loom.semaphore_give %54 : memref<1x64x128xf16>
            loom.semaphore_give %48 : memref<1x32x64xf16>
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%38, %29, %70 : memref<1x32x128xf16>, memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%29 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
              %73 = arith.mulf %in_5, %in_6 : f16
              %74 = arith.addf %in, %73 : f16
              linalg.yield %74 : f16
            }
            loom.semaphore_give %51 : memref<1x32x32xf16>
            loom.semaphore_give %38 : memref<1x32x128xf16>
            linalg.copy ins(%40 : memref<1x32x1xf16>) outs(%36 : memref<1x32x1xf16>)
            loom.semaphore_give %40 : memref<1x32x1xf16>
          }
          loom.semaphore_give %25 : memref<1x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %36 : memref<1x32x1xf16>, memref<1x32x1xf16>) outs(%35 : memref<1x32x1xf16>) {
          ^bb0(%in: f16, %in_3: f16, %out: f16):
            %65 = math.log %in : f16
            %66 = arith.addf %65, %in_3 : f16
            linalg.yield %66 : f16
          }
          loom.semaphore_give %36 : memref<1x32x1xf16>
          %55 = loom.broadcast ins(%33 : memref<1x32x1xf16>) outs(%50 : memref<1x32x32xf16>) dim(2) -> memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>
          loom.semaphore_give %33 : memref<1x32x1xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%29, %55 : memref<1x32x128xf16>, memref<1x32x128xf16, strided<[?, ?, ?], offset: ?>>) outs(%24 : memref<1x32x128xf16>) {
          ^bb0(%in: f16, %in_3: f16, %out: f16):
            %65 = arith.divf %in, %in_3 : f16
            linalg.yield %65 : f16
          }
          loom.semaphore_give %50 : memref<1x32x32xf16>
          loom.semaphore_give %29 : memref<1x32x128xf16>
          %56 = loom.alloc [8, 1, 32, 1] on @L1 : memref<8x1x32x1xf16>
          %57 = loom.semaphore_take %56 : memref<8x1x32x1xf16> -> memref<8x1x32x1xf16>
          %58 = loom.semaphore_take %34 : memref<1x32x1xf16> -> memref<1x32x1xf16>
          loom.sync ins(%35 : memref<1x32x1xf16>) outs(%58 : memref<1x32x1xf16>)
          loom.gather ins(%58 : memref<1x32x1xf16>) outs(%57 : memref<8x1x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
          loom.semaphore_give %58 : memref<1x32x1xf16>
          loom.semaphore_give %35 : memref<1x32x1xf16>
          %59 = loom.alloc [8, 1, 32, 128] on @L1 : memref<8x1x32x128xf16>
          %60 = loom.semaphore_take %59 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
          %61 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
          loom.sync ins(%24 : memref<1x32x128xf16>) outs(%61 : memref<1x32x128xf16>)
          loom.gather ins(%61 : memref<1x32x128xf16>) outs(%60 : memref<8x1x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
          loom.semaphore_give %61 : memref<1x32x128xf16>
          loom.semaphore_give %24 : memref<1x32x128xf16>
          %62 = loom.alloc [8, 1, 32, 32] on @L1 : memref<8x1x32x32xf16>
          %63 = loom.semaphore_take %62 : memref<8x1x32x32xf16> -> memref<8x1x32x32xf16>
          %64 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %64 {
            linalg.fill ins(%cst_1 : f16) outs(%32 : memref<1x32x1xf16>)
            %65 = loom.semaphore_take %56 : memref<8x1x32x1xf16> -> memref<8x1x32x1xf16>
            loom.sync ins(%57 : memref<8x1x32x1xf16>) outs(%65 : memref<8x1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%65 : memref<8x1x32x1xf16>) outs(%32 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %69 = arith.maximumf %in, %out : f16
              linalg.yield %69 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%65, %32 : memref<8x1x32x1xf16>, memref<1x32x1xf16>) outs(%57 : memref<8x1x32x1xf16>) {
            ^bb0(%in: f16, %in_4: f16, %out: f16):
              %69 = arith.subf %in, %in_4 : f16
              %70 = math.exp %69 : f16
              linalg.yield %70 : f16
            }
            loom.semaphore_give %32 : memref<1x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%31 : memref<1x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%57 : memref<8x1x32x1xf16>) outs(%31 : memref<1x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %69 = arith.addf %in, %out : f16
              linalg.yield %69 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%57, %31 : memref<8x1x32x1xf16>, memref<1x32x1xf16>) outs(%57 : memref<8x1x32x1xf16>) {
            ^bb0(%in: f16, %in_4: f16, %out: f16):
              %69 = arith.divf %in, %in_4 : f16
              linalg.yield %69 : f16
            }
            loom.semaphore_give %31 : memref<1x32x1xf16>
            %66 = loom.broadcast ins(%57 : memref<8x1x32x1xf16>) outs(%63 : memref<8x1x32x32xf16>) dim(3) -> memref<8x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
            loom.semaphore_give %65 : memref<8x1x32x1xf16>
            loom.semaphore_give %57 : memref<8x1x32x1xf16>
            %67 = loom.semaphore_take %59 : memref<8x1x32x128xf16> -> memref<8x1x32x128xf16>
            loom.sync ins(%60 : memref<8x1x32x128xf16>) outs(%67 : memref<8x1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%67, %66 : memref<8x1x32x128xf16>, memref<8x1x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>) outs(%60 : memref<8x1x32x128xf16>) {
            ^bb0(%in: f16, %in_4: f16, %out: f16):
              %69 = arith.mulf %in, %in_4 : f16
              linalg.yield %69 : f16
            }
            loom.semaphore_give %63 : memref<8x1x32x32xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<1x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%60 : memref<8x1x32x128xf16>) outs(%23 : memref<1x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %69 = arith.addf %in, %out : f16
              linalg.yield %69 : f16
            }
            loom.semaphore_give %67 : memref<8x1x32x128xf16>
            loom.semaphore_give %60 : memref<8x1x32x128xf16>
            %reinterpret_cast_3 = memref.reinterpret_cast %arg2 to offset: [%26], sizes: [1, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            %68 = loom.semaphore_take %22 : memref<1x32x128xf16> -> memref<1x32x128xf16>
            loom.sync ins(%23 : memref<1x32x128xf16>) outs(%68 : memref<1x32x128xf16>)
            loom.copy %68, %reinterpret_cast_3 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<1x32x128xf16> to memref<1x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %68 : memref<1x32x128xf16>
            loom.semaphore_give %23 : memref<1x32x128xf16>
          }
        } {loom.block_sym = @tile_b, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
}
