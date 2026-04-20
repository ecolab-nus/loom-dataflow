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
    func.func @flash_decode__x8_y8__d0i1_d1i0__f01__dim_x_level0_bc8_n_n_dim_x_level0_bc8__tile_b64__tile_n512__tile_s64(%arg0: memref<16x128x8192xf16>, %arg1: memref<16x8192x128xf16>, %arg2: memref<16x32x128xf16>, %arg3: memref<16x32x128xf16>) {
      %c8192 = arith.constant 8192 : index
      %c67108864 = arith.constant 67108864 : index
      %c262144 = arith.constant 262144 : index
      %c16 = arith.constant 16 : index
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
        scf.for %arg6 = %c0 to %c16 step %c1 {
          %20 = arith.muli %arg6, %c8 overflow<nsw> : index
          %21 = arith.addi %arg5, %20 : index
          %22 = loom.alloc [64, 32, 128] on @L1 : memref<64x32x128xf16>
          %23 = loom.semaphore_take %22 : memref<64x32x128xf16> -> memref<64x32x128xf16>
          %24 = loom.semaphore_take %22 : memref<64x32x128xf16> -> memref<64x32x128xf16>
          %25 = arith.muli %arg4, %c262144 : index
          %reinterpret_cast = memref.reinterpret_cast %arg3 to offset: [%25], sizes: [64, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<64x32x128xf16, strided<[4096, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast, %24 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<64x32x128xf16, strided<[4096, 128, 1], offset: ?>> to memref<64x32x128xf16>
          %26 = arith.muli %21, %c64 : index
          %27 = loom.alloc [64, 32, 128] on @L1 : memref<64x32x128xf16>
          %28 = loom.semaphore_take %27 : memref<64x32x128xf16> -> memref<64x32x128xf16>
          linalg.fill ins(%cst : f16) outs(%28 : memref<64x32x128xf16>)
          %29 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
          %30 = loom.semaphore_take %29 : memref<64x32x1xf16> -> memref<64x32x1xf16>
          %31 = loom.semaphore_take %29 : memref<64x32x1xf16> -> memref<64x32x1xf16>
          %32 = loom.semaphore_take %29 : memref<64x32x1xf16> -> memref<64x32x1xf16>
          %33 = loom.semaphore_take %29 : memref<64x32x1xf16> -> memref<64x32x1xf16>
          linalg.fill ins(%cst_0 : f16) outs(%33 : memref<64x32x1xf16>)
          %34 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
          %35 = loom.semaphore_take %34 : memref<64x32x1xf16> -> memref<64x32x1xf16>
          linalg.fill ins(%cst_1 : f16) outs(%35 : memref<64x32x1xf16>)
          %36 = loom.alloc [64, 32, 128] on @L1 : memref<64x32x128xf16>
          %37 = loom.semaphore_take %36 : memref<64x32x128xf16> -> memref<64x32x128xf16>
          %38 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
          %39 = loom.semaphore_take %38 : memref<64x32x1xf16> -> memref<64x32x1xf16>
          %40 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
          %41 = loom.semaphore_take %40 : memref<64x32x1xf16> -> memref<64x32x1xf16>
          %42 = loom.alloc [64, 32, 1] on @L1 : memref<64x32x1xf16>
          %43 = loom.semaphore_take %42 : memref<64x32x1xf16> -> memref<64x32x1xf16>
          %44 = loom.alloc [64, 128, 512] on @L1 : memref<64x128x512xf16>
          %45 = loom.semaphore_take %44 : memref<64x128x512xf16> -> memref<64x128x512xf16>
          %46 = loom.alloc [64, 32, 512] on @L1 : memref<64x32x512xf16>
          %47 = loom.semaphore_take %46 : memref<64x32x512xf16> -> memref<64x32x512xf16>
          %48 = loom.alloc [64, 512, 128] on @L1 : memref<64x512x128xf16>
          %49 = loom.semaphore_take %48 : memref<64x512x128xf16> -> memref<64x512x128xf16>
          %50 = arith.muli %arg4, %c67108864 : index
          %51 = arith.addi %50, %26 : index
          %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%51], sizes: [64, 128, 512], strides: [1048576, 8192, 1] : memref<16x128x8192xf16> to memref<64x128x512xf16, strided<[1048576, 8192, 1], offset: ?>>
          loom.copy %reinterpret_cast_3, %45 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<64x128x512xf16, strided<[1048576, 8192, 1], offset: ?>> to memref<64x128x512xf16>
          linalg.fill ins(%cst : f16) outs(%47 : memref<64x32x512xf16>)
          linalg.batch_matmul ins(%24, %45 : memref<64x32x128xf16>, memref<64x128x512xf16>) outs(%47 : memref<64x32x512xf16>)
          loom.semaphore_give %45 : memref<64x128x512xf16>
          linalg.fill ins(%cst_1 : f16) outs(%39 : memref<64x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x32x512xf16>) outs(%39 : memref<64x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %61 = arith.maximumf %in, %out : f16
            linalg.yield %61 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<64x32x1xf16>, memref<64x32x1xf16>) outs(%39 : memref<64x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %61 = arith.mulf %in_5, %cst_2 : f16
            %62 = arith.cmpf ogt, %in, %61 : f16
            %63 = arith.select %62, %in, %61 : f16
            linalg.yield %63 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%47, %39 : memref<64x32x512xf16>, memref<64x32x1xf16>) outs(%47 : memref<64x32x512xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %61 = arith.mulf %in, %cst_2 : f16
            %62 = arith.subf %61, %in_5 : f16
            %63 = math.exp %62 : f16
            linalg.yield %63 : f16
          }
          linalg.fill ins(%cst : f16) outs(%41 : memref<64x32x1xf16>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%47 : memref<64x32x512xf16>) outs(%41 : memref<64x32x1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %61 = arith.addf %in, %out : f16
            linalg.yield %61 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%35, %39 : memref<64x32x1xf16>, memref<64x32x1xf16>) outs(%43 : memref<64x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %61 = arith.subf %in, %in_5 : f16
            %62 = math.exp %61 : f16
            linalg.yield %62 : f16
          }
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %43, %41 : memref<64x32x1xf16>, memref<64x32x1xf16>, memref<64x32x1xf16>) outs(%33 : memref<64x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %61 = arith.mulf %in, %in_5 : f16
            %62 = arith.addf %61, %in_6 : f16
            linalg.yield %62 : f16
          }
          loom.semaphore_give %41 : memref<64x32x1xf16>
          %52 = arith.muli %21, %c8192 : index
          %53 = arith.addi %50, %52 : index
          %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%53], sizes: [64, 512, 128], strides: [1048576, 128, 1] : memref<16x8192x128xf16> to memref<64x512x128xf16, strided<[1048576, 128, 1], offset: ?>>
          loom.copy %reinterpret_cast_4, %49 src_mem_space @mem_DRAM dst_mem_space @mem_L1, broadcast : [1, 1] region : (UL : [%arg5, %arg4], LR : [%arg5, %arg4]) : memref<64x512x128xf16, strided<[1048576, 128, 1], offset: ?>> to memref<64x512x128xf16>
          linalg.fill ins(%cst : f16) outs(%37 : memref<64x32x128xf16>)
          linalg.batch_matmul ins(%47, %49 : memref<64x32x512xf16>, memref<64x512x128xf16>) outs(%37 : memref<64x32x128xf16>)
          loom.semaphore_give %49 : memref<64x512x128xf16>
          loom.semaphore_give %47 : memref<64x32x512xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%37, %28, %43 : memref<64x32x128xf16>, memref<64x32x128xf16>, memref<64x32x1xf16>) outs(%28 : memref<64x32x128xf16>) {
          ^bb0(%in: f16, %in_5: f16, %in_6: f16, %out: f16):
            %61 = arith.mulf %in_5, %in_6 : f16
            %62 = arith.addf %in, %61 : f16
            linalg.yield %62 : f16
          }
          loom.semaphore_give %43 : memref<64x32x1xf16>
          loom.semaphore_give %37 : memref<64x32x128xf16>
          linalg.copy ins(%39 : memref<64x32x1xf16>) outs(%35 : memref<64x32x1xf16>)
          loom.semaphore_give %39 : memref<64x32x1xf16>
          loom.semaphore_give %24 : memref<64x32x128xf16>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%33, %35 : memref<64x32x1xf16>, memref<64x32x1xf16>) outs(%32 : memref<64x32x1xf16>) {
          ^bb0(%in: f16, %in_5: f16, %out: f16):
            %61 = math.log %in : f16
            %62 = arith.addf %61, %in_5 : f16
            linalg.yield %62 : f16
          }
          loom.semaphore_give %33 : memref<64x32x1xf16>
          loom.semaphore_give %35 : memref<64x32x1xf16>
          %54 = loom.alloc [128, 64, 32, 1] on @L1 : memref<128x64x32x1xf16>
          %55 = loom.semaphore_take %54 : memref<128x64x32x1xf16> -> memref<128x64x32x1xf16>
          %56 = loom.semaphore_take %29 : memref<64x32x1xf16> -> memref<64x32x1xf16>
          loom.sync ins(%32 : memref<64x32x1xf16>) outs(%56 : memref<64x32x1xf16>)
          loom.gather ins(%56 : memref<64x32x1xf16>) outs(%55 : memref<128x64x32x1xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
          loom.semaphore_give %56 : memref<64x32x1xf16>
          loom.semaphore_give %32 : memref<64x32x1xf16>
          %57 = loom.alloc [128, 64, 32, 128] on @L1 : memref<128x64x32x128xf16>
          %58 = loom.semaphore_take %57 : memref<128x64x32x128xf16> -> memref<128x64x32x128xf16>
          %59 = loom.semaphore_take %27 : memref<64x32x128xf16> -> memref<64x32x128xf16>
          loom.sync ins(%28 : memref<64x32x128xf16>) outs(%59 : memref<64x32x128xf16>)
          loom.gather ins(%59 : memref<64x32x128xf16>) outs(%58 : memref<128x64x32x128xf16>) across(%arg5 : index) region : (UL : [%c0, %arg4], LR : [%c7, %arg4])
          loom.semaphore_give %59 : memref<64x32x128xf16>
          loom.semaphore_give %28 : memref<64x32x128xf16>
          %60 = arith.cmpi eq, %arg5, %c0 : index
          scf.if %60 {
            linalg.fill ins(%cst_1 : f16) outs(%31 : memref<64x32x1xf16>)
            %61 = loom.semaphore_take %54 : memref<128x64x32x1xf16> -> memref<128x64x32x1xf16>
            loom.sync ins(%55 : memref<128x64x32x1xf16>) outs(%61 : memref<128x64x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%61 : memref<128x64x32x1xf16>) outs(%31 : memref<64x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %64 = arith.maximumf %in, %out : f16
              linalg.yield %64 : f16
            }
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%61, %31 : memref<128x64x32x1xf16>, memref<64x32x1xf16>) outs(%55 : memref<128x64x32x1xf16>) {
            ^bb0(%in: f16, %in_6: f16, %out: f16):
              %64 = arith.subf %in, %in_6 : f16
              %65 = math.exp %64 : f16
              linalg.yield %65 : f16
            }
            loom.semaphore_give %31 : memref<64x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%30 : memref<64x32x1xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%55 : memref<128x64x32x1xf16>) outs(%30 : memref<64x32x1xf16>) {
            ^bb0(%in: f16, %out: f16):
              %64 = arith.addf %in, %out : f16
              linalg.yield %64 : f16
            }
            %62 = loom.semaphore_take %57 : memref<128x64x32x128xf16> -> memref<128x64x32x128xf16>
            loom.sync ins(%58 : memref<128x64x32x128xf16>) outs(%62 : memref<128x64x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%62, %55, %30 : memref<128x64x32x128xf16>, memref<128x64x32x1xf16>, memref<64x32x1xf16>) outs(%58 : memref<128x64x32x128xf16>) {
            ^bb0(%in: f16, %in_6: f16, %in_7: f16, %out: f16):
              %64 = arith.divf %in_6, %in_7 : f16
              %65 = arith.mulf %in, %64 : f16
              linalg.yield %65 : f16
            }
            loom.semaphore_give %61 : memref<128x64x32x1xf16>
            loom.semaphore_give %55 : memref<128x64x32x1xf16>
            loom.semaphore_give %30 : memref<64x32x1xf16>
            linalg.fill ins(%cst : f16) outs(%23 : memref<64x32x128xf16>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%58 : memref<128x64x32x128xf16>) outs(%23 : memref<64x32x128xf16>) {
            ^bb0(%in: f16, %out: f16):
              %64 = arith.addf %in, %out : f16
              linalg.yield %64 : f16
            }
            loom.semaphore_give %62 : memref<128x64x32x128xf16>
            loom.semaphore_give %58 : memref<128x64x32x128xf16>
            %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%25], sizes: [64, 32, 128], strides: [4096, 128, 1] : memref<16x32x128xf16> to memref<64x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            %63 = loom.semaphore_take %22 : memref<64x32x128xf16> -> memref<64x32x128xf16>
            loom.sync ins(%23 : memref<64x32x128xf16>) outs(%63 : memref<64x32x128xf16>)
            loom.copy %63, %reinterpret_cast_5 src_mem_space @mem_L1 dst_mem_space @mem_DRAM, broadcast : [8, 1] region : (UL : [%c0, %arg4], LR : [%c7, %arg4]) : memref<64x32x128xf16> to memref<64x32x128xf16, strided<[4096, 128, 1], offset: ?>>
            loom.semaphore_give %63 : memref<64x32x128xf16>
            loom.semaphore_give %23 : memref<64x32x128xf16>
          }
        } {loom.block_sym = @tile_s, loom.iter_type = #loom.iter_type<temporal>}
        scf.reduce 
      } {loom.iter_types = [#loom.iter_type<spatial>, #loom.iter_type<spatial>], loom.logical_levels = [0, 0], loom.physical_dims = [@dim_y, @dim_x]}
      return
    }
  }
}
